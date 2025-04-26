# Dataset image counts
fine_grained_datasets = {
    "DTD": 1692,
    "Flowers": 2463,
    "Cars": 8041,
    "Aircraft": 3333,
    "Pets": 3669,
    "Caltech101": 2465,
    "UCF101": 3783,
    "Eurosat": 8100,

}

imagenet_datasets = {
    "ImageNet_A": 7500,
    "ImageNet_R": 30000,
    "ImageNet_K": 50889,
    "ImageNet_V": 10000,
    "ImageNet_I": 50000,
}

# DTD timing info
dtd_images = fine_grained_datasets["DTD"]
dtd_time_minutes = 25

# Time per image
time_per_image = dtd_time_minutes / dtd_images

# Total images across all datasets
# total_images = sum(datasets.values())
total_fine_grained_images = sum(fine_grained_datasets.values())
total_imagenet_images = sum(imagenet_datasets.values())

# Total time fg
total_fg_time_minutes = total_fine_grained_images * time_per_image
total_fg_time_hours = total_fg_time_minutes / 60
total_fg_time_days = total_fg_time_hours / 24

# Total time for ImageNet datasets
total_imagenet_time_minutes = total_imagenet_images * time_per_image
total_imagenet_time_hours = total_imagenet_time_minutes / 60
total_imagenet_time_days = total_imagenet_time_hours / 24


# Output
print(f"Total time for fine-grained datasets: {total_fg_time_days:.2f} days")
print(f"Total time for ImageNet datasets: {total_imagenet_time_days:.2f} days")

# Output the time for each dataset
for dataset, count in fine_grained_datasets.items():
    time_minutes = count * time_per_image
    time_hours = time_minutes / 60
    time_days = time_hours / 24
    print(f"{dataset}: {time_days:.2f} days")

# Output the time for each ImageNet dataset
for dataset, count in imagenet_datasets.items():
    time_minutes = count * time_per_image
    time_hours = time_minutes / 60
    time_days = time_hours / 24
    print(f"{dataset}: {time_days:.2f} days")

# Output the total time for all datasets
print(f"Total time for all datasets: {total_fg_time_days + total_imagenet_time_days:.2f} days")
