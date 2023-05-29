set -e

script_dir=$(cd `dirname $0` && pwd)
container_name="dolphin-documentation-builder"
image_name="dolphin-documentation-builder"
image_tag="latest"

docker run -it \
           -v "${script_dir}/..":'/workspace/' \
           -w /workspace/ \
           --name ${container_name} \
           ${image_name}:${image_tag} \
           bash