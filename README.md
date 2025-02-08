# Used Tools

- miniconda
- python version 3.12.x


## Create Environment from package-list.txt

```sh
> conda create --name bookrecommender --file package-list.txt
> conda activate bookrecommender
> jupyter notebook --notebook-dir="<project_folder_full_path>" --no-browser --allow-root
```

## Create Environment Manually

```sh
> conda create --name bookrecommender
> conda activate bookrecommender
> jupyter notebook --notebook-dir="<project_folder_full_path>" --no-browser --allow-root
```

`conda install conda-forge::kagglehub`
`conda install conda-forge::pandas`
`conda install conda-forge::matplotlib`
`conda install conda-forge::seaborn`
`conda install conda-forge::python-dotenv`
`conda install conda-forge::langchain-community`
`pip install -qU langchain-ollama`
`conda install conda-forge::langchain-chroma`
`conda install conda-forge::transformers`
`pip install gradio`
`conda install conda-forge::notebook`
`conda install conda-forge::ipywidgets`
`conda install conda-forge::python-language-server`
`conda install conda-forge::python-lsp-server`
`pip install --upgrade huggingface_hub`
`pip install huggingface_hub[tensorflow]`
`pip install tensorflow`
`pip install tf-keras`

## Optional
`conda install conda-forge::langchain-openai`

## VS Code Extensions
- Python (Microsoft)
- Jupyter (Microsoft)
- Data Wrangler (Microsoft)
