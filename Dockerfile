FROM ubuntu:18.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH
ARG TAG

RUN apt-get update && \
    apt-get install -y autoconf gcc gfortran g++ make wget gsl-bin git libgsl-dev && apt-get clean all
    
RUN apt-get install -y curl grep sed dpkg && \
       TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
       curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
       dpkg -i tini.deb && \
       rm tini.deb && \
       apt-get clean

RUN /usr/sbin/groupadd -g 1000 user && \
       /usr/sbin/useradd -u 1000 -g 1000 -d /opt/redmapper redmapper && \
	mkdir /opt/redmapper && chown redmapper.user /opt/redmapper && \
	chown redmapper.user /opt
USER redmapper

RUN wget --quiet http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
	rm ~/miniconda.sh
RUN . /opt/conda/etc/profile.d/conda.sh && conda create --yes --name redmapper-env
RUN	echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
        echo "conda activate redmapper-env" >> ~/.bashrc
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/startup.sh && \
        echo "conda activate redmapper-env" >> ~/startup.sh

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && conda install --yes python=3.8 numpy scipy nose && \
       conda clean -af --yes

LABEL redmapper-tag="${TAG}"

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && cd ~/ && \
       git clone https://github.com/erykoff/redmapper --branch=${TAG} && cd ~/redmapper && pip install -r requirements.txt --no-cache-dir && \
       cd ~/redmapper && python setup.py install

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash", "-lc" ]
