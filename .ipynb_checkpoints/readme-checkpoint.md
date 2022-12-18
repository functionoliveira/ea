# Docker commands
docker ps -> list all containers running
docker build - < dockerfile -t environment -> this command will build our environment
docker run -d -t environment -> this command will execute the docker image
docker exec -it <docker_image_id> /bin/bash -> use this command if you need access the docker

# Docker Compose
docker-compose up -d -> this command will build and run the dockerfile, remove '-d' if you want to see the training output
