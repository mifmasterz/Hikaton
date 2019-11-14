using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using System.Threading.Tasks;
using GrabBusinessProfile.Models;
using Microsoft.ML.Data;
using Microsoft.ML;
using Microsoft.ML.Trainers.LightGbm;

namespace GrabBusinessProfile.Controllers
{
    [Produces("application/json")]
    [Route("api/[controller]")]

    public class PredictionController : Controller
    {
        private static string MODEL_FILEPATH = @"MLModel.zip";
        private static string MODEL2_FILEPATH = @"MLModel2.zip";
        public PredictionController()
        {

        }
       
        static ModelOutput PredictPrice(ModelInput input)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            string modelPath = MODEL_FILEPATH;
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(mlModel);

            // Use model to make prediction on input data
            ModelOutput result = predEngine.Predict(input);
            return result;
        }

        static ModelOutput2 PredictSaving(ModelInput2 input)
        {

            // Create new MLContext
            MLContext mlContext = new MLContext();

            // Load model & create prediction engine
            string modelPath = MODEL2_FILEPATH;
            ITransformer mlModel = mlContext.Model.Load(modelPath, out var modelInputSchema);
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput2, ModelOutput2>(mlModel);

            // Use model to make prediction on input data
            ModelOutput2 result = predEngine.Predict(input);
            return result;
        }

        [HttpGet("[action]")]
        public async Task<ActionResult> PredictSaving(string Day1, float Distance, int Hour, string VehicleType)
        {
            var hasil = new OutputData() { IsSucceed = true };

            var modelInput = new ModelInput2() { Day = Day1, Vehicle_Type = VehicleType, Hour = Hour, Ride_Distance__km_ = Distance };
            try
            {
                var datas = PredictSaving(modelInput);
                hasil.Data = datas;
            }
            catch (Exception ex)
            {
                hasil.IsSucceed = false;
                hasil.ErrorMessage = ex.Message;
            }

            return Ok(hasil);
        }

        [HttpGet("[action]")]
        public async Task<ActionResult> PredictPrice(string Day1, float Distance, int Hour,string VehicleType)
        {
            var hasil = new OutputData() { IsSucceed = true };

            var modelInput = new ModelInput() { Day= Day1, Vehicle_Type = VehicleType, Hour = Hour, Ride_Distance__km_ = Distance  };
            try
            {
                var datas = PredictPrice(modelInput);
                hasil.Data = datas;
            }
            catch (Exception ex)
            {
                hasil.IsSucceed = false;
                hasil.ErrorMessage = ex.Message;
            }

            return Ok(hasil);
        }
    }
    public class ModelInput
    {
        [ColumnName("Ride Distance (km)"), LoadColumn(0)]
        public float Ride_Distance__km_ { get; set; }


        [ColumnName("Vehicle Type"), LoadColumn(1)]
        public string Vehicle_Type { get; set; }


        [ColumnName("Promo Value"), LoadColumn(2)]
        public float Promo_Value { get; set; }


        [ColumnName("Fare"), LoadColumn(3)]
        public float Fare { get; set; }


        [ColumnName("Paid"), LoadColumn(4)]
        public float Paid { get; set; }


        [ColumnName("Hour"), LoadColumn(5)]
        public float Hour { get; set; }


        [ColumnName("Minute"), LoadColumn(6)]
        public float Minute { get; set; }


        [ColumnName("Class"), LoadColumn(7)]
        public string Class { get; set; }


        [ColumnName("Day"), LoadColumn(8)]
        public string Day { get; set; }


        [ColumnName("Saving"), LoadColumn(9)]
        public float Saving { get; set; }


    }

    public class ModelOutput
    {
        // ColumnName attribute is used to change the column name from
        // its default value, which is the name of the field.
        [ColumnName("PredictedLabel")]
        public String Prediction { get; set; }
        public float[] Score { get; set; }
    }
    public class ModelInput2
    {
        [ColumnName("Ride Distance (km)"), LoadColumn(0)]
        public float Ride_Distance__km_ { get; set; }


        [ColumnName("Vehicle Type"), LoadColumn(1)]
        public string Vehicle_Type { get; set; }


        [ColumnName("Promo Value"), LoadColumn(2)]
        public float Promo_Value { get; set; }


        [ColumnName("Fare"), LoadColumn(3)]
        public float Fare { get; set; }


        [ColumnName("Paid"), LoadColumn(4)]
        public float Paid { get; set; }


        [ColumnName("Hour"), LoadColumn(5)]
        public float Hour { get; set; }


        [ColumnName("Minute"), LoadColumn(6)]
        public float Minute { get; set; }


        [ColumnName("Class"), LoadColumn(7)]
        public string Class { get; set; }


        [ColumnName("Day"), LoadColumn(8)]
        public string Day { get; set; }


        [ColumnName("Saving"), LoadColumn(9)]
        public float Saving { get; set; }


    }
    public class ModelOutput2
    {
        public float Score { get; set; }
    }
}