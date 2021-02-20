FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ARG TAG

RUN apt-get update && \
    apt-get install -y autoconf gcc gfortran g++ make wget gsl-bin git libgsl-dev && apt-get clean all
    
RUN apt-get install -y curl grep sed dpkg
RUN TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
        curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb
RUN dpkg -i tini.deb
RUN rm tini.deb
RUN apt-get clean

RUN /usr/sbin/groupadd -g 1000 user && \
       /usr/sbin/useradd -u 1000 -g 1000 -d /opt/redmapper redmapper && \
	mkdir /opt/redmapper && chown redmapper.user /opt/redmapper && \
	chown redmapper.user /opt
USER redmapper

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/download/4.9.2-5/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
        /bin/bash ~/miniforge.sh -b -p /opt/conda && \
	rm ~/miniforge.sh
RUN . /opt/conda/etc/profile.d/conda.sh && conda create --yes --name redmapper-env
RUN	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        echo "conda activate redmapper-env" >> ~/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/startup.sh && \
        echo "conda activate redmapper-env" >> ~/startup.sh

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && \
       conda install --yes python=3.8 numpy scipy astropy matplotlib pyyaml fitsio esutil healpy healsparse && \
       conda clean -af --yes

LABEL redmapper-tag="${TAG}"

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && cd ~/ && \
       git clone https://github.com/erykoff/redmapper --branch=${TAG} && cd ~/redmapper && \
       cd ~/redmapper && python setup.py install

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash", "-lc" ]
