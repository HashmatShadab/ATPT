"""
Robust Test-time Prompt Tuning (RTPT) for CLIP models.

This script implements test-time adaptation techniques for CLIP models to improve
their robustness against distribution shifts and adversarial attacks. It uses prompt
tuning to adapt the model at test time without modifying the model weights.
"""

import argparse
import logging
import time
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    # Fallback for older torchvision versions
    BICUBIC = Image.BICUBIC

from open_clip.custom_openai_clip import get_coop as get_coop_openai
from open_clip.custom_openai_clip import get_open_clip
from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from data.cls_to_names import flower102_classes, food101_classes, dtd_classes, caltech101_classes, pets_classes, \
    sun397_classes, cars_classes, ucf101_classes, aircraft_classes, eurosat_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, set_random_seed
from utils.logger import setup_logger
import os

import torchattacks
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np

openai_model_dict = {
    "delta_clip_l14_224": "hf-hub:zw123/delta_clip_l14_224",
    "tecoa4": "hf-hub:chs20/tecoa4-clip",
    "tecoa2": "hf-hub:chs20/tecoa2-clip",
    "fare2": "hf-hub:chs20/fare2-clip",
    "fare4": "hf-hub:chs20/fare4-clip",
    "RN50": "RN50",
}


def plot_image(imgs, title=None, nrow=4, figsize=(12, 8)):
    """
    Plots a single image or a list of images in a grid.

    Args:
        imgs: Single torch.Tensor, PIL.Image, or numpy.ndarray, or a list of them.
        title: Optional string for the plot title.
        nrow: Number of images per row if plotting a list.
        figsize: Size of the figure for multiple images.
    """

    def prepare(img):
        """Helper to convert Tensor/PIL/numpy to numpy format for plotting."""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.dim() == 3:
                img = img.permute(1, 2, 0)
            img = img.numpy()
        elif isinstance(img, Image.Image):
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")
        return img

    if isinstance(imgs, list):
        imgs = [prepare(img) for img in imgs]
        n_imgs = len(imgs)
        ncols = nrow
        nrows = (n_imgs + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten()

        for idx, ax in enumerate(axes):
            if idx < n_imgs:
                ax.imshow(imgs[idx])
                ax.axis('off')
            else:
                ax.remove()  # Remove extra empty subplots

        if title:
            fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    else:
        img = prepare(imgs)
        plt.imshow(img)
        plt.axis('off')
        if title:
            plt.title(title)
        plt.show()



def get_top_sim(sim_matrix, args):
    """
    Calculate the mean similarity of top-k most similar samples for each sample.

    Args:
        sim_matrix (torch.Tensor): Similarity matrix between samples.
        args (argparse.Namespace): Arguments containing the top_k parameter.

    Returns:
        torch.Tensor: Mean similarity scores of top-k neighbors for each sample.
    """
    # Exclude self-similarity (which is 1.0) by setting it to negative infinity
    sim_matrix[sim_matrix>=1.0] = float('-inf')
    # Get top-k similarity values for each sample
    top_k_values, _ = sim_matrix.topk(args.top_k, dim=-1) # default is 20 neighbors
    # Calculate mean similarity
    top_k_mean = top_k_values.mean(dim=-1)
    return top_k_mean

def print_args(args):
    """
    Format command line arguments for printing.

    Args:
        args (argparse.Namespace): Command line arguments.

    Returns:
        str: Formatted string of all arguments.
    """
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += f"{arg}:{content}\n"
    return s


def calculate_entropy(outputs):
    """
    Calculate entropy for each sample in the batch.

    Args:
        outputs (torch.Tensor): Model output logits.

    Returns:
        torch.Tensor: Entropy for each sample.
    """
    return -(outputs.softmax(1) * outputs.log_softmax(1)).sum(1)

def entropy_avg(outputs):
    """
    Calculate the average entropy of model outputs.

    Args:
        outputs (torch.Tensor): Model output logits.

    Returns:
        torch.Tensor: Mean entropy across all samples.
    """
    # Calculate entropy for each sample and return mean
    return calculate_entropy(outputs).mean()

def select_confident_samples(logits, top):
    """
    Select the most confident samples based on entropy.

    Lower entropy indicates higher confidence in the prediction.

    Args:
        logits (torch.Tensor): Model output logits.
        top (float): Proportion of samples to select (0.0 to 1.0).

    Returns:
        tuple: (selected_logits, selected_indices)
    """
    # Calculate entropy for each sample in the batch
    batch_entropy = calculate_entropy(logits)
    # Select indices of samples with lowest entropy (highest confidence)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def test_time_tuning(model, inputs, optimizer, scaler, args, logger=None):
    """
    Perform test-time tuning of the model using entropy minimization.

    This function adapts the model at test time by minimizing the entropy of predictions
    on the input batch. It selects confident samples based on their entropy and uses
    them for adaptation.

    Args:
        model (torch.nn.Module): The model to be tuned.
        inputs (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision training.
        args (argparse.Namespace): Arguments containing tuning parameters.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        None
    """
    # Track indices of confident samples
    selected_idx = None

    if logger:
        logger.debug(f"Starting test-time tuning with {args.tta_steps} steps")

    # Perform test-time adaptation for specified number of steps
    for j in range(args.tta_steps):
        # Forward pass
        output = model(inputs)

        # Use only confident samples for adaptation
        if selected_idx is not None:
            # Use previously selected confident samples
            output = output[selected_idx]
        else:
            # Select confident samples based on entropy
            output, selected_idx = select_confident_samples(output, args.selection_p)
            if logger:
                logger.debug(f"Selected {len(selected_idx)}/{inputs.size(0)} samples for adaptation")

        # Calculate loss as average entropy (lower is better)
        loss = entropy_avg(output)

        if logger and (j == 0 or j == args.tta_steps - 1 or j % 5 == 0):
            logger.debug(f"Step {j+1}/{args.tta_steps}, Loss: {loss.item():.6f}")

        # Update model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if logger:
        logger.debug(f"Completed test-time tuning with final loss: {loss.item():.6f}")

    return


def main():
    IMAGENET_CLASSNAMES = (
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
        "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
        "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
        "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander",
        "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog",
        "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin",
        "box turtle", "banded gecko", "green iguana", "Carolina anole",
        "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard",
        "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile",
        "American alligator", "triceratops", "worm snake", "ring-necked snake",
        "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake",
        "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra",
        "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake",
        "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider",
        "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider",
        "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl",
        "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet",
        "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck",
        "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby",
        "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch",
        "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
        "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
        "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron",
        "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot",
        "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher",
        "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion",
        "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel",
        "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle",
        "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound",
        "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound",
        "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound",
        "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier",
        "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier",
        "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier",
        "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier",
        "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer",
        "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier",
        "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier",
        "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever",
        "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla",
        "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
        "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
        "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
        "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
        "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
        "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
        "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
        "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
        "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
        "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
        "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle",
        "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
        "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
        "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
        "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger",
        "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose",
        "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle",
        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper",
        "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper",
        "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly",
        "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly",
        "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit",
        "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse",
        "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
        "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)",
        "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat",
        "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
        "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque",
        "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin",
        "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey",
        "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda",
        "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
        "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown",
        "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
        "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle",
        "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo",
        "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel",
        "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel",
        "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)",
        "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini",
        "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet",
        "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra",
        "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest",
        "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe",
        "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton",
        "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran",
        "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw",
        "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking",
        "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker",
        "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard",
        "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
        "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
        "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
        "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table",
        "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig",
        "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar",
        "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder",
        "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute",
        "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed",
        "freight car", "French horn", "frying pan", "fur coat", "garbage truck",
        "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola",
        "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine",
        "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer",
        "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet",
        "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar",
        "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep",
        "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat",
        "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library",
        "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion",
        "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag",
        "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask",
        "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone",
        "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile",
        "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor",
        "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa",
        "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail",
        "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina",
        "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart",
        "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush",
        "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench",
        "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case",
        "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube",
        "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
        "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag",
        "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho",
        "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
        "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
        "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
        "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator",
        "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser",
        "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal",
        "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard",
        "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
        "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
        "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
        "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
        "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
        "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
        "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
        "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa",
        "submarine", "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge",
        "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe",
        "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball",
        "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof",
        "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store",
        "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod",
        "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard",
        "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling",
        "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball",
        "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink",
        "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle",
        "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing",
        "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website",
        "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu",
        "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
        "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
        "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
        "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
        "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
        "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
        "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef",
        "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player",
        "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn",
        "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
        "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"
    )

    # Parse arguments and set random seed
    args = parser.parse_args()
    set_random_seed(args.seed)

    # Calculate alpha from epsilon if not provided
    args.alpha = args.eps / args.alpha_eps_ratio

    # Create output directory path with experiment parameters
    args.output_dir = os.path.join(args.output_dir, args.arch, args.test_sets)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging

    # Create a log name that includes TTA variations
    # Format floating point values and ensure filename is valid
    log_name = f"ADV_eps_{args.eps}_steps_{args.steps}_TPT_lr_{args.lr}_step_{args.tta_steps}_selection_{args.selection_p}_topk_neighbours_{args.top_k}_sftemp_{args.softmax_temp}"
    logger, log_file = setup_logger(log_name, args.output_dir, level=logging.INFO)
    logger.info(print_args(args))

    # Ensure GPU is available
    assert args.gpu is not None
    set_random_seed(args.seed)
    logger.info(f"Use GPU: {args.gpu} for training")

    # Determine class names based on dataset
    dset = args.test_sets
    if len(dset) > 1:
        # For multi-character dataset names (e.g., 'Caltech101')
        # This would require importing the specific classes for each dataset
        # For now, we keep using eval for this case as it's not a common path
        classnames = eval(f"{dset.lower()}_classes")
    else:
        # For single-character dataset codes (ImageNet variants)
        assert dset in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes

        # Select appropriate class names based on dataset code
        if dset == 'A':
            # ImageNet-A
            classnames = [classnames_all[i] for i in imagenet_a_mask]
        elif dset == 'R':
            # ImageNet-R
            classnames = [classnames_all[i] for i, m in enumerate(imagenet_r_mask) if m]
        elif dset == 'V':
            # ImageNet-V
            classnames = [classnames_all[i] for i in imagenet_v_mask]
        else:
            # For ImageNet (I) or ImageNet-K
            classnames = classnames_all
    args.classnames = classnames

    # Initialize model with CoOp (Context Optimization)
    # model,_,preprocess_openclip,tokenizer = get_open_clip(args.arch, args.gpu)
    # Initialize model with CoOp (Context Optimization)
    if args.arch in openai_model_dict:
        actual_model_name = openai_model_dict[args.arch]
        model, _, preprocess_openclip, tokenizer = get_open_clip(actual_model_name, args.gpu)
    else:
        model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)

    # move model to GPU
    model = model.cuda(args.gpu)
    #model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)
    prompts = [f"a photo of a {c}." for c in classnames]
    tokenized_prompts = tokenizer(prompts, context_length=40)
    # move tokneized prompts to GPU
    tokenized_prompts = tokenized_prompts.cuda(args.gpu)

    model_state = None



    logger.info(f"=> Model created: visual backbone {args.arch}")

    # Move model to GPU
    if not torch.cuda.is_available():
        logger.warning('Using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)



    # Set up additional training parameters
    scaler = None  # No mixed precision scaling used
    cudnn.benchmark = not args.no_cudnn_benchmark  # Enable cudnn benchmarking for faster training unless disabled, default is True



    batchsize = 16 # Process images one at a time for test-time adaptation

    # Create dataset and data loader
    val_dataset = build_dataset(dset, preprocess_openclip, args.data, mode=args.dataset_mode)
    logger.info(f"Number of test samples: {len(val_dataset)}")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False,
                num_workers=args.workers, pin_memory=not args.no_pin_memory)

    logger.info(f"Evaluating dataset: {dset}")

    optimizer=None
    optim_state=None
    # Run evaluation with test-time adaptation
    results = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, preprocess_openclip, logger, tokenized_prompts)

    # Clean up to free memory
    del val_dataset, val_loader

    # Format and save results
    if args.eps <= 0:
        # Clean accuracy (no adversarial attack)
        log_msg = f"=> Acc. on testset [{dset}]: Clean Acc @1 {results[0]}/ TTA Clean Acc @1 {results[1]}"
        save_log = {'clean_acc': results[0], 'tta_clean_acc': results[1]}
    else:
        # Adversarial accuracy
        log_msg = f"=> Acc. on testset [{dset}]: Adv Acc @1 {results[0]}/ TTA Adv Acc @1 {results[1]}"
        save_log = {'adv_acc': results[0], 'tta_adv_acc': results[1]}

    # Log results
    logger.info(log_msg)

    # Save results to file
    torch.save(save_log, os.path.join(args.output_dir, 'results_log.pt'))


def get_adversarial_image(image, target, attack, path, index, output_dir, logger=None):
    """
    Generate or load a cached adversarial image.

    Args:
        image (torch.Tensor): Original image tensor.
        target (torch.Tensor): Target label.
        attack (torchattacks.Attack): Adversarial attack object.
        path (list or None): Path to the original image file.
        index (int): Index of the current sample.
        output_dir (str): Directory to save/load adversarial images.
        logger (logging.Logger, optional): Logger for logging information.

    Returns:
        PIL.Image.Image: Adversarial image.
    """
    # Create a unique filename for the adversarial image
    if path is not None:
        # Extract filename from path and the preceding directory
        img_filename = os.path.basename(path[0])
        # change the extension to .png
        img_filename = os.path.splitext(img_filename)[0] + '.png'
        parent_folder_name = os.path.basename(os.path.dirname(path[0]))
        adv_img_path = os.path.join(output_dir, f"{parent_folder_name}_{img_filename}")

    else:
        # If path is not available, use index as identifier
        adv_img_path = os.path.join(output_dir, f"{index}.png")

    # Check if adversarial image already exists
    if os.path.exists(adv_img_path):
        if logger:
            logger.info(f"Loading existing adversarial image from {adv_img_path}")
        # Load existing adversarial image
        img_adv = Image.open(adv_img_path).convert('RGB')
    else:
        # Create adversarial image using attack
        adv_image = attack(image, target)
        if logger:
            logger.info(f"Generated adversarial image with shape: {adv_image.shape}")

        img_adv = transforms.ToPILImage()(adv_image.squeeze(0))
        # Save the adversarial image
        img_adv.save(adv_img_path)
        if logger:
            logger.info(f"Saved adversarial image to {adv_img_path}")

    return img_adv


def test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args, data_transform, logger=None, tokenized_prompts=None):
    """
    Evaluate model performance with test-time adaptation.

    This function evaluates the model on a validation dataset, applying test-time adaptation
    to improve performance. It can also evaluate robustness against adversarial attacks
    if specified in the arguments.

    Args:
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        model (torch.nn.Module): The model to evaluate.
        model_state (dict, optional): Model state dictionary for resetting.
        optimizer (torch.optim.Optimizer): Optimizer for test-time tuning.
        optim_state (dict): Optimizer state dictionary for resetting.
        scaler (torch.cuda.amp.GradScaler, optional): Gradient scaler for mixed precision.
        args (argparse.Namespace): Arguments containing evaluation parameters.
        data_transform (callable): Data transformation function.

    Returns:
        list: [original_accuracy, test_time_adapted_accuracy]
    """
    # Initialize metrics tracking
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)  # Original model accuracy
    tpt1 = AverageMeter('TTAAcc@1', ':6.2f', Summary.AVERAGE)  # Test-time adapted accuracy
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    # Progress display
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, tpt1],
        prefix='Test: ')

    # Set model to evaluation mode
    model.eval()

    if logger:
        logger.info(f"Starting evaluation with batch size: {args.batch_size}, selection percentage: {args.selection_p}")
        logger.info(f"Test-time adaptation steps: {args.tta_steps}, learning rate: {args.lr}")



    end = time.time()
    # Create directory for saving adversarial images if needed


    # Iterate through validation data
    for i, data in enumerate(val_loader):
        # Handle different return formats (with or without path)
        if len(data) == 3:
            images, target, path = data
        else:
            images, target = data
            path = None

        assert args.gpu is not None
        target = target.cuda(args.gpu, non_blocking=True)
        images = images.cuda(args.gpu, non_blocking=True)


        # Get original model outputs and features
        with torch.no_grad():
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokenized_prompts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            clip_output = (100.0 * image_features @ text_features.T) ##.softmax(dim=-1)
            # get the top 1 prediction value and index
            # _, clip_pred = clip_output.topk(1, dim=1)
            # print(f"clip_pred: {clip_pred}")

        # Measure accuracy
        acc1, acc5 = accuracy(clip_output, target, topk=(1, 5))  # Original model accuracy
        # tpt_acc1, _ = accuracy(tta_output, target, topk=(1, 5))  # Test-time adapted accuracy

        # Update accuracy metrics
        top1.update(acc1[0], images.size(0))
        # tpt1.update(tpt_acc1[0], images.size(0))

        if logger and (i < 5 or i % 20 == 0):  # Log detailed info for first few samples and periodically
            logger.debug(f"Sample {i+1}: Original accuracy: {acc1[0].item():.2f}")


        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if (i+1) % args.print_freq == 0 or (i+1) == len(val_loader):
            if logger:
                if args.eps <= 0:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, clip_acc1={top1.avg}')
                else:
                    logger.info(f'iter:{i+1}/{len(val_loader)}, clip_adv1={top1.avg}')
            progress.display(i)

    # Display final summary
    progress.display_summary()

    if logger:
        if args.eps <= 0:
            logger.info(f"Final results - Original accuracy: {top1.avg:.2f}, TTA accuracy: {tpt1.avg:.2f}")
            logger.info(f"Improvement from TTA: {tpt1.avg - top1.avg:.2f}")
        else:
            logger.info(f"Final results - Adversarial accuracy: {top1.avg:.2f}, TTA adversarial accuracy: {tpt1.avg:.2f}")
            logger.info(f"Improvement from TTA: {tpt1.avg - top1.avg:.2f}")

    # Return original and test-time adapted accuracies
    return [top1.avg, tpt1.avg]


if __name__ == '__main__':
    # Set up command-line argument parser
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')

    # Dataset parameters
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='Caltech101',
                        help='Dataset to evaluate on (e.g., Caltech101, A, R, K, V, I for ImageNet variants)')
    parser.add_argument('--dataset_mode', type=str, default='test',
                        help='Dataset split to use (train, val, test)')

    # Model parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50',
                        help='Model architecture (RN50, ViT-B/32, tecoa4, tecoa2, fare2, fare4, delta_clip_l14_224 etc.)')
    parser.add_argument('--resolution', default=224, type=int,
                        help='CLIP image resolution')

    # Hardware and performance parameters
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    # pin memory, default is True
    parser.add_argument('--no_pin_memory', action='store_true',
                        help='Pin memory for data loading')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N',
                        help='Mini-batch size for augmentation')
    parser.add_argument('-p', '--print-freq', default=20, type=int, metavar='N',
                        help='Print frequency (default: 200)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use')
    parser.add_argument('--no_cudnn_benchmark', action='store_true',
                        help='Disable cudnn benchmarking for potentially more deterministic behavior')

    # Prompt tuning parameters
    parser.add_argument('--n_ctx', default=4, type=int,
                        help='Number of tunable context tokens')
    parser.add_argument('--ctx_init', default=None, type=str,
                        help='Initial values for tunable prompts')

    # Experiment parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducibility')
    parser.add_argument('--output_dir', type=str, default='output_results/ckps/rtpt',
                        help='Directory to save results')

    # Adversarial attack parameters
    parser.add_argument('--eps', default=0.0, type=float,
                        help='Epsilon for adversarial attack (0.0 for clean evaluation)')
    parser.add_argument('--alpha', default=0.0, type=float,
                        help='Step size for adversarial attack (if not provided, calculated as eps/alpha_eps_ratio)')
    parser.add_argument('--alpha_eps_ratio', default=4.0, type=float,
                        help='Ratio of epsilon to alpha when alpha is not explicitly provided (default: 4.0)')
    parser.add_argument('--steps', type=int, default=0,
                        help='Number of steps for adversarial attack')

    # Test-time adaptation parameters
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float, metavar='LR',
                        help='Learning rate for test-time adaptation', dest='lr')
    parser.add_argument('--selection_p', default=0.1, type=float,
                        help='Proportion of confident samples to select for adaptation (0.0-1.0)')
    parser.add_argument('--tta_steps', default=1, type=int,
                        help='Number of test-time adaptation steps')
    parser.add_argument('--top_k', default=20, type=int,
                        help='Number of neighbors for similarity calculation')
    parser.add_argument('--softmax_temp', default=0.01, type=float,
                        help='Temperature parameter for softmax in similarity weighting')

    # Pre-trained model parameters
    parser.add_argument('--load_tecoa', type=str, default='',
                        choices=['', 'RN50-eps1', 'ViT-B/32-eps1', 'ViT-B/32-eps4'],
                        help='Load robust vision encoder (TeCoA)')

    # Run the main function
    main()
