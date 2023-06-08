import wget


base_url = "https://g-fa6726.9601a.bd7c.data.globus.org/DeepBlueData_bg257f92t/S"

for i in range(1, 50):
    if i < 10:
        wget.download(base_url + "0" + str(i) + ".mat", out="../data/umich")
    else:
        wget.download(base_url + str(i) + ".mat", out="../data/umich")