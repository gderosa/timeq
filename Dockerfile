FROM blang/latex:ubuntu

# Minted + Pygments
RUN apt-get update
RUN apt-get install -y biber
