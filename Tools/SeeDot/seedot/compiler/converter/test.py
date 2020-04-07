# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import math
import numpy as np

def bar(a, b):
    print(a, b)

def foo(*args):
    bar(*args)

foo(1, 2)
