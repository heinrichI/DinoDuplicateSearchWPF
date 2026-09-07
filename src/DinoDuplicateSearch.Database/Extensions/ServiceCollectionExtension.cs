using DinoDuplicateSearch.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace DinoDuplicateSearch.Database.Extensions;

public static class ServiceCollectionExtension
{
    public static void AddDatabase(this IServiceCollection services)
    {
        services.AddSingleton<IFeatureCache, FeatureCache>();
    }
}
