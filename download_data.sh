# Bash script to download and unpack the data needed to reproduce the results.
# Allows some time to get a response from the request since the file is pretty big. It may take a few minutes!
#
# You can also do this manually, if you want.
# 1. Go to https://surfdrive.surf.nl/files/index.php/s/DtU5rW7za4DeNb5
# 2. Click on download and download the zip file (it's almost 4!)
# 3. Unpack the file and rename the main folder to data

wget  https://surfdrive.surf.nl/files/index.php/s/DtU5rW7za4DeNb5/download

unzip download

mv SI_Pagerank_data/ data
