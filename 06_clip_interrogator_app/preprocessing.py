import os, subprocess

print("Download preprocessed cache files...")
CACHE_URLS = [
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-L-14_openai_trendings.pkl',
    # 'ViT-L-14/openai'
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_artists.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_flavors.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_mediums.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_movements.pkl',
    'https://huggingface.co/pharma/ci-preprocess/resolve/main/ViT-H-14_laion2b_s32b_b79k_trendings.pkl',
]
os.makedirs('cache', exist_ok=True)
for url in CACHE_URLS:
    print(subprocess.run(['wget', url, '-P', 'cache'], stdout=subprocess.PIPE).stdout.decode('utf-8'))

print('llegamos hasta acá')



