#!/bin/bash
set -e

TARGET_DIR=../../pymvg-website

# remove old versions
rm -rf ${TARGET_DIR}/*

# copy new version
cp -a build/singlehtml/* ${TARGET_DIR}
touch ${TARGET_DIR}/.nojekyll

# git stuff
cd ${TARGET_DIR}
git add .
git commit -m "update"
git push origin gh-pages
