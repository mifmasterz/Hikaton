﻿using System;
using System.Collections.Generic;
using System.Text;

namespace GrabBusinessProfile.Models
{
    public class OutputData
    {
        public bool IsSucceed { set; get; }
        public object Data { set; get; }
        public string ErrorMessage { set; get; }
    }
}
