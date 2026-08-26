#!/bin/bash
wget -O data/bar_crawl_data.zip "https://archive.ics.uci.edu/static/public/515/bar+crawl+detecting+heavy+drinking.zip"
unzip -q -o data/bar_crawl_data.zip -d data/
unzip -q -o data/data.zip -d data/
rm data/bar_crawl_data.zip
rm data/data.zip
