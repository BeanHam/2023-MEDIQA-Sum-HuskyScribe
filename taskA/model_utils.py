import os
from t5_io_utils import *
import json
from model_constants import  SECTION2FULL
import transformers

def convert_full2section(pred):
    full2section = { v.lower() :k for k,v in SECTION2FULL.items() }
    return full2section.get(pred.lower().replace('_',' '), pred.lower()).lower()