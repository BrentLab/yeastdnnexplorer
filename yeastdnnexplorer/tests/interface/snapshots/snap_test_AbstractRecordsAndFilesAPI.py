# -*- coding: utf-8 -*-
# snapshottest: v1 - https://goo.gl/zC4yUc
from __future__ import unicode_literals

from snapshottest import Snapshot


snapshots = Snapshot()

snapshots['test_save_response_records_and_files 1'] = '''id,uploader_id,upload_date,modifier_id,modified_date,binding_id,promoter_id,background_id,fileformat_id,file
10690,1,2024-03-26,1,2024-03-26 14:28:43.825628+00:00,4079,4,6,5,promotersetsig/10690.csv.gz
10694,1,2024-03-26,1,2024-03-26 14:28:44.739775+00:00,4083,4,6,5,promotersetsig/10694.csv.gz
10754,1,2024-03-26,1,2024-03-26 14:29:01.837335+00:00,4143,4,6,5,promotersetsig/10754.csv.gz
10929,1,2024-03-26,1,2024-03-26 14:29:45.379790+00:00,4318,4,6,5,promotersetsig/10929.csv.gz
10939,1,2024-03-26,1,2024-03-26 14:29:47.853980+00:00,4327,4,6,5,promotersetsig/10939.csv.gz
'''
