# How to make model serving server to use docker container.


# download docker image

```dash
docker pull tensorflow/serving
```

# make serving server

```dash
docker run -p 8501:8501 --mount type=bind,source=$PWD/'modelfile',target=/models/'model_name' -e MODEL_NAME='model_name' -td tensorflow/serving
```
