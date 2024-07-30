import os
from setuptools import setup, find_packages

os.system('pip install '
          'PyYAML, '
          'llama-cpp-python, '
          'bitsandbytes==0.43.1, ')

setup(
    name="RuPersonaAgent",
    version="1.0.0",
    install_requires=[
        'typer==0.9.4',
        'aiohttp==3.8.5',
        'absl-py==1.3.0',
        'aiosignal==1.3.1',
        'antlr4-python3-runtime==4.8',
        'async-timeout==4.0.2',
        'black==22.10.0',
        'beautifulsoup4==4.12.2',
        'cachetools==5.2.0',
        'charset-normalizer==3.2.0',
        'clearml==1.8.1',
        'click==8.1.3',
        'commonmark==0.9.1',
        'contourpy==1.0.6',
        'cycler==0.11.0',
        'datasets==2.2.2',
        'dill==0.3.4',
        'docstring-parser==0.15',
        'fake-useragent==1.2.1',
        'filelock==3.8.0',
        'flake8==5.0.4',
        'fire==0.4.0',
        'fonttools==4.38.0',
        'frozenlist==1.3.3',
        'fsspec==2023.5.0',
        'furl==2.1.3',
        'google-auth==2.14.1',
        'google-auth-oauthlib==0.4.6',
        'grpcio==1.50.0',
        'huggingface-hub==0.23.2',
        'hydra-core==1.1.2',
        'idna==3.4',
        'importlib-resources==5.2',
        'jsonlines==3.1.0',
        'jsonargparse==4.17.0',
        'jsonschema==4.17.3',
        'h5py==3.9.0',
        'kiwisolver==1.4.4',
        'lightseq==3.0.1',
        'lightning-utilities==0.3.0',
        'markdown==3.3.2',
        'markupsafe==2.1.1',
        'matplotlib==3.6.2',
        'multidict==6.0.2',
        'multiprocess==0.70.12.2',
        'mypy-extensions==1.0.0',
        'numpy==1.21',
        'nest-asyncio==1.6.0',
        'nvidia-cublas-cu11==11.10.3.66',
        'nvidia-cuda-nvrtc-cu11==11.7.99',
        'nvidia-cuda-runtime-cu11==11.7.99',
        'nvidia-cudnn-cu11==8.5.0.96',
        'oauthlib==3.2.2',
        'omegaconf==2.1.2',
        'orderedmultidict==1.0.1',
        'packaging==23.1',
        'pathlib2==2.3.7.post1',
        'pathspec==0.10.2',
        'peft==0.11.1',
        'pillow==9.3.0',
        'platformdirs==2.5.4',
        'protobuf==3.19',
        'pytest==7.1.3',
        'psutil==5.9.4',
        'pymorphy2==0.9.1',
        'pyarrow==10.0.1',
        'pyasn1==0.4.8',
        'pyasn1-modules==0.2.8',
        'pygments==2.13.0',
        'pyjwt==2.4.0',
        'pytorch-pretrained-bert==0.6.2',
        'pyparsing==3.0.9',
        'pyrsistent==0.19.2',
        'python-dateutil==2.8.2',
        'pytorch-lightning==1.8.3.post1',
        'pytz==2022.6',
        'pyyaml==6.0',
        'regex==2022.10.31',
        'requests==2.31.0',
        'requests-oauthlib==1.3.1',
        'responses==0.18.0',
        'rich==13.7.1',
        'rsa==4.9',
        'six==1.16.0',
        'spacy==3.6.1',
        'scikit-learn==1.3.0',
        'tensorboard==2.11.0',
        'tensorboard-data-server==0.6.1',
        'tensorboard-plugin-wit==1.8.1',
        'tensorboardx==2.5.0',
        'tensorflow==2.11.0',
        'termcolor==2.1.0',
        'tokenizers==0.19',
        'tomli==1.2.3',
        'torch==1.13.0',
        'torch-tb-profiler==0.4.0',
        'torchmetrics==0.10.3',
        'tqdm==4.62.1',
        'transformers==4.43.3',
        'typing-extensions==4.7.1',
        'urllib3==1.26.19',
        'werkzeug==2.2.2',
        'xxhash==3.3.0',
        'yarl==1.8.1',
        'nltk==3.7',
        'scipy==1.10.1',
        'pyhamcrest==2.0.4',
        'sentencepiece==0.2.0',
        'langchain==0.2.9',
        'faiss-cpu==1.8.0',
        'tiktoken==0.7.0',
        'sentence-transformers',
        'lightning==1.8.6',
        'accelerate==0.31.0',
        'ctransformers==0.2.27',
        'unstructured==0.14.5',
        'evaluate==0.4.2',
        'pandas==1.5.1',
    ],
    packages=find_packages(include=[
        'generative_model', 'speech_extraction',
        'inference_optimization', 'internet_memory_model',
        'knowledge_distillation', 'personification',
        'rule_based_information_extraction',
        'tests', 'datasets'
    ])

)
