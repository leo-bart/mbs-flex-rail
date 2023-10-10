# -*- coding: utf-8 -*-

import MBS.BodyConnections.Forces
import numpy as np
import helper_funcs as hf
import gjk

class contact(MBS.BodyConnections.Forces.force):
    def __init__(self,name_='Contact force'):
        super().__init__(name_)