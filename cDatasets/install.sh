for i in {2..20}
do
    ck add dataset:adpcm-$(printf "%04d" $i) --tags=dataset,audio,adpcm --file=telecom_data/$i.adpcm
    ck add dataset:au-$(printf "%04d" $i) --tags=dataset,audio,au --file=telecom_gsm_data/$i.au
    ck add dataset:cdataset-dijkstra-$(printf "%04d" $i) --tags=dijkstra,dataset --file=network_dijkstra_data/$i.dat
    ck add dataset:cdataset-patricia-$(printf "%04d" $i) --tags=dataset,patricia --file=network_patricia_data/$i.udp
    ck add dataset:cdataset-qsort-$(printf "%04d" $i) --tags=dataset,qsort,sorting --file=automotive_qsort_data/$i.dat
    ck add dataset:image-jpeg-$(printf "%04d" $i) --tags=dataset,jpeg,image --file=consumer_jpeg_data/$i.jpg
    ck add dataset:image-pgm-$(printf "%04d" $i) --tags=dataset,image,pgm --file=automotive_susan_data/$i.pgm
    ck add dataset:image-ppm-$(printf "%04d" $i) --tags=dataset,ppm,image --file=consumer_jpeg_data/$i.ppm
    ck add dataset:image-tiff-$(printf "%04d" $i) --tags=dataset,image,tiff,tif,orig --file=consumer_tiff_data/$i.tif
    ck add dataset:image-tiff-$(printf "%04d" $i)-bw --tags=dataset,image,tiff,tif,bw --file=consumer_tiff_data/$i.bw.tif
    ck add dataset:image-tiff-$(printf "%04d" $i)-nocomp --tags=dataset,image,tiff,tif,nocomp --file=consumer_tiff_data/$i.nocomp.tif
    ck add dataset:pcm-$(printf "%04d" $i) --tags=dataset,audio,pcm --file=telecom_data/$i.pcm
done
