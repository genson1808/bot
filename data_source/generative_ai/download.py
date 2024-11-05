import os, wget

file_links = [
    {
        "title": "Flower Categorization using Deep Convolutional Neural Networks",
        "url": "https://arxiv.org/pdf/1708.03763",
    },
    {
        "title": "Stuck in the mangrove mud: The risk of trace element exposure to shore crabs in restored urban mangroves",
        "url": "https://research-repository.griffith.edu.au/server/api/core/bitstreams/0e69d343-7892-4e83-bc8c-3d82c7750352/content",
    },
    {
        "title": "Flight: A FaaS-Based Framework for Complex and Hierarchical Federated Learning",
        "url": "https://arxiv.org/pdf/2409.16495",
    },
    {
        "title": "Phase Separation Bursting and Symmetry Breaking inside an Evaporating Droplet; Formation of a Flower-like Pattern",
        "url": "https://arxiv.org/pdf/2409.07095",
    },
]

def is_exist(file_link):
    return os.path.exists(f"./{file_link['title']}.pdf")

for file_link in file_links:
    if not is_exist(file_link):
        wget.download(file_link["url"], out=f"./{file_link['title']}.pdf")

