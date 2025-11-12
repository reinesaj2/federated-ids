Free use of the Edge-IIoTset dataset for academic research purposes is hereby granted in perpetuity. Use for commercial purposes is allowable after asking the leader author, Dr Mohamed Amine Ferrag, who has asserted his right under the Copyright.

The details of the Edge-IIoT dataset were published in following the paper. For the academic/public use of these datasets, the authors have to cities the following paper:

Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning", TechRxiv, 2022, Doi: 10.36227/techrxiv.18857336.v1

Link to paper : https://www.techrxiv.org/articles/preprint/Edge-IIoTset_A_New_Comprehensive_Realistic_Cyber_Security_Dataset_of_IoT_and_IIoT_Applications_for_Centralized_and_Federated_Learning/18857336

********************************************

The directories of the Edge-IIoTset dataset include the following:

•File 1 (Normal traffic)

-File 1.1 (Distance): This file includes two documents, namely, Distance.csv and Distance.pcap. The IoT sensor (Ultrasonic sensor) is used to capture the IoT data.

-File 1.2 (Flame_Sensor): This file includes two documents, namely, Flame_Sensor.csv and Flame_Sensor.pcap. The IoT sensor (Flame Sensor) is used to capture the IoT data.

-File 1.3 (Heart_Rate): This file includes two documents, namely, Flame_Sensor.csv and Flame_Sensor.pcap. The IoT sensor (Flame Sensor) is used to capture the IoT data.

-File 1.4 (IR_Receiver): This file includes two documents, namely, IR_Receiver.csv and IR_Receiver.pcap. The IoT sensor (IR (Infrared) Receiver Sensor) is used to capture the IoT data.

-File 1.5 (Modbus): This file includes two documents, namely, Modbus.csv and Modbus.pcap. The IoT sensor (Modbus Sensor) is used to capture the IoT data.

-File 1.6 (phValue): This file includes two documents, namely, phValue.csv and phValue.pcap. The IoT sensor (pH-sensor PH-4502C) is used to capture the IoT data.

-File 1.7 (Soil_Moisture): This file includes two documents, namely, Soil_Moisture.csv and Soil_Moisture.pcap. The IoT sensor (Soil Moisture Sensor v1.2) is used to capture the IoT data.

-File 1.8 (Sound_Sensor): This file includes two documents, namely, Sound_Sensor.csv and Sound_Sensor.pcap. The IoT sensor (LM393 Sound Detection Sensor) is used to capture the IoT data.

-File 1.9 (Temperature_and_Humidity): This file includes two documents, namely, Temperature_and_Humidity.csv and Temperature_and_Humidity.pcap. The IoT sensor (DHT11 Sensor) is used to capture the IoT data.

-File 1.10 (Water_Level): This file includes two documents, namely, Water_Level.csv and Water_Level.pcap. The IoT sensor (Water sensor) is used to capture the IoT data.

•File 2 (Attack traffic): 

-File 2.1 (Attack traffic (CSV files)): This file includes 13 documents, namely, Backdoor_attack.csv, DDoS_HTTP_Flood_attack.csv, DDoS_ICMP_Flood_attack.csv, DDoS_TCP_SYN_Flood_attack.csv, DDoS_UDP_Flood_attack.csv, MITM_attack.csv, OS_Fingerprinting_attack.csv, Password_attack.csv, Port_Scanning_attack.csv, Ransomware_attack.csv, SQL_injection_attack.csv, Uploading_attack.csv, Vulnerability_scanner_attack.csv, XSS_attack.csv. Each document is specific for each attack.

-File 2.2 (Attack traffic (PCAP files)): This file includes 13 documents, namely, Backdoor_attack.pcap, DDoS_HTTP_Flood_attack.pcap, DDoS_ICMP_Flood_attack.pcap, DDoS_TCP_SYN_Flood_attack.pcap, DDoS_UDP_Flood_attack.pcap, MITM_attack.pcap, OS_Fingerprinting_attack.pcap, Password_attack.pcap, Port_Scanning_attack.pcap, Ransomware_attack.pcap, SQL_injection_attack.pcap, Uploading_attack.pcap, Vulnerability_scanner_attack.pcap, XSS_attack.pcap. Each document is specific for each attack.

•File 3 (Selected dataset for ML and DL): 

-File 3.1 (DNN-EdgeIIoT-dataset): This file contains a selected dataset for the use of evaluating deep learning-based intrusion detection systems. 

-File 3.2 (ML-EdgeIIoT-dataset): This file contains a selected dataset for the use of evaluating traditional machine learning-based intrusion detection systems. 

********************************************

Step 1: Downloading The Edge-IIoTset dataset From the Kaggle platform
from google.colab import files

!pip install -q kaggle

files.upload()

!mkdir ~/.kaggle

!cp kaggle.json ~/.kaggle/

!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot -f "Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"

!unzip DNN-EdgeIIoT-dataset.csv.zip

!rm DNN-EdgeIIoT-dataset.csv.zip

Step 2: Reading the Datasets' CSV file to a Pandas DataFrame:
import pandas as pd

import numpy as np

df = pd.read_csv('DNN-EdgeIIoT-dataset.csv', low_memory=False) 

 Step 3 : Exploring some of the DataFrame's contents:
df.head(5)

print(df['Attack_type'].value_counts())

Step 4: Dropping data (Columns, duplicated rows, NAN, Null..):
from sklearn.utils import shuffle

drop_columns = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4","arp.dst.proto_ipv4", 

         "http.file_data","http.request.full_uri","icmp.transmit_timestamp",

         "http.request.uri.query", "tcp.options","tcp.payload","tcp.srcport",

         "tcp.dstport", "udp.port", "mqtt.msg"]

df.drop(drop_columns, axis=1, inplace=True)

df.dropna(axis=0, how='any', inplace=True)

df.drop_duplicates(subset=None, keep="first", inplace=True)

df = shuffle(df)

df.isna().sum()

print(df['Attack_type'].value_counts())
Step 5: Categorical data encoding (Dummy Encoding):
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing

def encode_text_dummy(df, name):

    dummies = pd.get_dummies(df[name])

    for x in dummies.columns:

        dummy_name = f"{name}-{x}"

        df[dummy_name] = dummies[x]

    df.drop(name, axis=1, inplace=True)

encode_text_dummy(df,'http.request.method')

encode_text_dummy(df,'http.referer')

encode_text_dummy(df,"http.request.version")

encode_text_dummy(df,"dns.qry.name.len")

encode_text_dummy(df,"mqtt.conack.flags")

encode_text_dummy(df,"mqtt.protoname")

encode_text_dummy(df,"mqtt.topic")

Step 6: Creation of the preprocessed dataset
df.to_csv('preprocessed_DNN.csv', encoding='utf-8')
********************************************

For more information about the dataset, please contact the leader author of this project, Dr Mohamed Amine Ferrag, on his email: mohamed.amine.ferrag@gmail.com  or  ferrag.mohamedamine@univ-guelma.dz .

More information about Dr. Mohamed Amine Ferrag is available at:

https://sites.google.com/site/mohamedamineferrag/ 

https://www.linkedin.com/in/Mohamed-Amine-Ferrag 

https://dblp.uni-trier.de/pid/142/9937.html 

https://www.researchgate.net/profile/Mohamed_Amine_Ferrag 

https://scholar.google.fr/citations?user=IkPeqxMAAAAJ&hl=fr&oi=ao 

https://www.scopus.com/authid/detail.uri?authorId=56115001200 

https://publons.com/researcher/1322865/mohamed-amine-ferrag/ 

https://orcid.org/0000-0002-0632-3172 

 

Last Updated: 18 Mar. 2022