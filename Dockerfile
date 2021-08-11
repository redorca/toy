FROM toy:latest

VOLUME /toy
WORKDIR /toy

CMD chmod -R 777 /toy

CMD ["/bin/bash"]