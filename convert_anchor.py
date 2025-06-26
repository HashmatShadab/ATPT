import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import math
import logging
import os

def Rot_map(V):
    assert len(V.shape) == 1
    assert np.linalg.norm(V) - 1 < 1e-8
    n_dim = V.shape[0]
    Rot = np.eye(n_dim)
    Rot_inv = np.eye(n_dim)
    for rotate in range(n_dim-1):
        rot_mat = np.eye(n_dim)
        rot_norm = np.sqrt(V[rotate]**2 + V[rotate+1]**2)
        cos_theta = V[rotate+1]/rot_norm
        sin_theta = V[rotate]/rot_norm
        rot_mat[rotate,rotate] = cos_theta
        rot_mat[rotate,rotate+1] = - sin_theta
        rot_mat[rotate+1,rotate] = sin_theta
        rot_mat[rotate+1,rotate+1] = cos_theta

        V = np.dot(rot_mat, V)

        Rot = np.dot(rot_mat, Rot)
        Rot_inv = np.dot(Rot_inv,rot_mat.transpose())
    return Rot, Rot_inv

def convert(text_features, downstream_feature = None, dim = 512, logger=None):
    '''text_features: Tensor, [N, dim]'''
    # Use default logger if none provided
    if logger is None:
        logger = logging.getLogger('convert_anchor')

    text_features = text_features /  text_features.norm(dim=-1, keepdim=True)
    logger.info(f"original cos similarity: {torch.mean(torch.matmul(text_features, text_features.T))}")
    anchor = torch.mean(text_features, dim = 0)
    anchor = anchor / torch.norm(anchor)
    anchor = anchor.detach().cpu().numpy()

    anchor = anchor.astype(np.float64)
    anchor = anchor / np.linalg.norm(anchor)
    target = np.zeros(dim)
    target[0] += 1.0

    R_0, R_0_inv = Rot_map(target)
    R_X, _ = Rot_map(np.dot(R_0, anchor))
    R = np.dot(np.dot(R_0_inv, R_X), R_0)
    R = torch.from_numpy(R).to(text_features.device)

    new_text_features = torch.matmul(text_features.double(), R.T)
    anchor = torch.from_numpy(anchor).to(R.device)
    target = torch.matmul(anchor, R.T)

    similarity = new_text_features.matmul(target.T)
    mincos = torch.min(similarity)
    theta = torch.arccos(mincos)

    theta1 = torch.arccos(new_text_features[:,0])
    theta2 = 2 * math.pi * (theta1) / ((theta) * 4)

    converted = torch.zeros(new_text_features.shape).double().to(new_text_features.device)
    converted[:,1:] = new_text_features[:,1:] * torch.sin(theta2).unsqueeze(1) / torch.sin(theta1).unsqueeze(1)
    converted[:,0] = torch.cos(theta2)

    converted = torch.matmul(converted, R)

    logger.info(f"converted cos similarity: {torch.mean(torch.matmul(converted, converted.T))}")
    if downstream_feature is not None:
        downstream_feature = downstream_feature /  downstream_feature.norm(dim=-1, keepdim=True)
        new_downstream_feature = torch.matmul(downstream_feature.double(), R.T)
        theta1 = torch.arccos(new_downstream_feature[:,0])
        theta2 = 2 * math.pi * (theta1) / ((theta) * 4)

        downstream_converted = torch.zeros(new_downstream_feature.shape).double().to(new_downstream_feature.device)
        downstream_converted[:,1:] = new_downstream_feature[:,1:] * torch.sin(theta2).unsqueeze(1) / torch.sin(theta1).unsqueeze(1)
        downstream_converted[:,0] = torch.cos(theta2)

        downstream_converted = torch.matmul(downstream_converted, R)

        return downstream_converted


    return converted

if __name__ == "__main__":
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert anchor features for CLIP model.')
    parser.add_argument('--base_filename', type=str, default="dtd", help='Base filename for input and output files (without extensions),')
    parser.add_argument('--dataset_size', '-ds', type=int, default=None,
                        help='Optional dataset size parameter for splitting features')
    parser.add_argument('--dim', type=int, default=512,
                        help='[DEPRECATED] Dimension of the feature vectors - now automatically detected from loaded features')
    parser.add_argument('--output_suffix', type=str, default='_clip_weight_a.npy',
                        help='Suffix for output filename (default: _clip_weight_a.npy)')

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    dataset_name = args.base_filename
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{dataset_name}_convert_anchor.log')

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger('convert_anchor')

    # Load input file
    input_file = args.base_filename + "_anchors.npy"
    output_file = args.base_filename + args.output_suffix

    logger.info(f"Loading features from {input_file}")
    weight = np.load(input_file)
    text_features = torch.from_numpy(weight)

    # Get dimension from loaded text_features
    feature_dim = text_features.shape[1]
    logger.info(f"Using feature dimension: {feature_dim} from loaded text_features")

    # Process features
    if args.dataset_size is None:
        logger.info(f"Converting all features")
        converted_features = convert(text_features, dim=feature_dim, logger=logger).float().numpy()
    else:
        logger.info(f"Converting features with dataset size: {args.dataset_size}")
        feat1 = convert(text_features[:args.dataset_size], dim=feature_dim, logger=logger)
        feat2 = convert(text_features[:args.dataset_size], text_features[args.dataset_size:], dim=feature_dim, logger=logger)
        converted_features = torch.cat((feat1, feat2)).float().numpy()
        logger.info(f"Output shape: {converted_features.shape}")

    # Save output
    logger.info(f"Saving converted features to {output_file}")
    np.save(output_file, converted_features)
    # converted = convert(text_features, text_features[:50])
    # converted2 = convert(text_features)[:50]
    # print(torch.sum(torch.abs(converted - converted2)))
    # print(converted2)
