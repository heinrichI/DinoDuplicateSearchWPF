using System.IO;
using Microsoft.Extensions.DependencyInjection;

namespace DinoDuplicateSearch.CV.Extensions;

public static class ServiceCollectionExtension
{
    public static void AddComputerVision(this IServiceCollection services)
    {
        services.AddSingleton<ISuperPointLightGluePipeline>(sp =>
        {
            var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "superpoint_lightglue_pipeline.onnx");
            return new SuperPointLightGluePipeline(path);
        });
    }
}
