#!/bin/bash

#
# get_data.sh: Downloads data from the *-*~~InTeRnEt~~*-*.
#
# (Specifically, downloads larger blobs like GloVe embeddings that we don't want to
# save in the git repo because it's bad form. (And GitHub probably won't let us.))
#
# Note to interested reader: I first wrote this to get data "from first principles:"
# e.g., download the entirety of GloVe, then preprocess it to pull out only the word
# embeddings we need. This seemed more transparent. But then I realized that anyone who
# is running this is probably capable of pulling numbers out of a text file and would
# rather things run fast than "from first principles." So we're just going to be pulling
# cached numpy arrays. This *should* be fine for most people, though is a bit less
# portable. (Though honestly, even that is probably up for debate, as folks on Windows
# won't be able to run this alltogether, and if Python and numpy play nicely, it might
# be easier for them to just grab those blobs.)
#
# This script aims to be idempotent, so you can run it multiple times if you're missing
# something.
#

# Print all commands as we go along. This is nice so you can see what's happening
# because you're running a script you downloaded from the Internet. And written by me.
# And I might be craaaazy! But I'm not. (Well, probably.) I mean, I could just turn this
# off somewhere (it'd be `set +x`) and do nefarious stuff, but nah, come on, let's just
# get some data.
set -x

# Exit on error. Also nice.
set -e

# First stop: GloVe town.
mkdir -p data/glove/
if [ ! -f data/glove/vocab-pc.glove.840B.300d.txt.npz ]; then
    curl https://homes.cs.washington.edu/~mbforbes/physical-commonsense/vocab-pc.glove.840B.300d.txt.npz > data/glove/vocab-pc.glove.840B.300d.txt.npz
fi

# Next up: Dependency Embeddings.
mkdir -p data/dep-embs/
if [ ! -f data/dep-embs/vocab-pc.dep-embs.npz ]; then
    curl https://homes.cs.washington.edu/~mbforbes/physical-commonsense/vocab-pc.dep-embs.npz > data/dep-embs/vocab-pc.dep-embs.npz
fi

# Then: ELMo.
mkdir -p data/elmo/
if [ ! -f data/elmo/sentences.elmo.npz ]; then
    curl https://homes.cs.washington.edu/~mbforbes/physical-commonsense/sentences.elmo.npz > data/elmo/sentences.elmo.npz
fi

# Make some more directories we'll need.
mkdir -p data/results/graphs/
