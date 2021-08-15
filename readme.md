# Demo on reverse image search based on pretrained model
---

## Installation:

- Dotnet framework (Tested on 4.8)

Can down load at 

- Miniconda3 - With python version 3.8.5
Just download this and install https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Windows-x86_64.exe or go to https://docs.conda.io/en/latest/miniconda.html for detailed installation guide

**Notes:**

We should make sure that installation path is ``C:\Users\<user_name>\miniconda3\`` as it is configured in image-indexer/image-indexer.csproj

In ``image-indexer/Program.cs``, we can see

```
var pythonPath = @"C:\Users\world\miniconda3;" +
            @"C:\Users\world\miniconda3\Scripts;" +
            @"C:\Users\world\miniconda3\Library\bin;" +
            @"C:\Users\world\miniconda3\Library;C:\Users\world\miniconda3\Library\mingw-w64\bin;";
var pythonHome = @"C:\Users\world\miniconda3";
var pythonLib = @"C:\Users\world\miniconda3\Lib;";
```

In which, ``C:\Users\world\miniconda3`` is a path pointing to miniconda installed folder. Path structure should be ``C:\Users\<user-name>\miniconda3``, of course user-name may vary

- Faiss

Quick install with miniconda:

```
conda install -c pytorch faiss-cpu
```

Github: https://github.com/facebookresearch/faiss

Document: https://faiss.ai/

Bonus, we can also use ``Spotify/Annoy`` for indexing, but it does not support removal of indexes:

*Can reference to https://github.com/spotify/annoy for annoy index, but currently not used*

- Supporting libraries:

Should install other neccessary libraries of python:

```
tensorflow
numpy
matplotlib
```

Simply type 1 command to install:

```
conda install numpy matplotlib tensorflow
```

## Run

```
cd image-indexer
dotnet run
```

Or 

Just run project ``image-indexer/`` in Vs

## Ideas behind the code 

- First, use a pretrained model as a feature extractor (e.g vgg16 in the code), which could be faster than yolo v3, feature map then will be flatten to an array of floats

- Use ``faiss`` as a vector indexer which could resolves nearest neighbor in a quick and memory efficient manner

- We could also use ``annoy`` (previously used in the code but removed) for indexing vectors, but due to its implementation, it won't support removal of indexes once saved to file