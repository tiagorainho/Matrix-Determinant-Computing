FROM --platform=linux/amd64 spagnuolocarmine/docker-mpi

#RUN useradd -ms /bin/bash project

#USER project
WORKDIR /project

RUN mkdir bin

COPY dataset dataset
COPY scripts scripts
COPY src src

WORKDIR /project/scripts
RUN ./build.sh

ENTRYPOINT ["sleep", "infinity" ]
