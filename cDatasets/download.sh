url_prefix="https://sourceforge.net/projects/cbenchmark/files/cDatasets/V1.1/cDatasets_V1.1_"
files=(
    "consumer_tiff_data.tar.gz"
    "office_data.tar.gz"
    "telecom_data.tar.gz"
    "consumer_jpeg_data.tar.gz"
    "telecom_gsm_data.tar.gz"
    "consumer_data.tar.gz"
    "bzip2_data.tar.gz"
    "network_patricia_data.tar.gz"
    "network_dijkstra_data.tar.gz"
    "automotive_susan_data.tar.gz"
    "automotive_qsort_data.tar.gz")
for file in "${files[@]}"
do
    echo "Downloading $file"
    url="${url_prefix}${file}"
    curl -L "$url" > "$file"
    tar -xzf "$file"
    rm "$file"
done
