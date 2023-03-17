# Dolphin


## Build

```
docker build  -f dockerfile \
              --rm \
              -t pytorch-research:latest \
              .
```

## Run

```
docker run \
        -it \
        --rm \
        --gpus all \
        -v "$(cd `dirname $0` && pwd)":"/app" \
        pytorch-research:latest \
        bash
```


