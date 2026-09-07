namespace DinoDuplicateSearch.CV;

public class GeometricVerifierFactory : IGeometricVerifierFactory
{
    private readonly LightGlueGeometricVerifier _lightGlue;
    private readonly SiftGeometricVerifier _sift;

    public GeometricVerifierFactory(LightGlueGeometricVerifier lightGlue, SiftGeometricVerifier sift)
    {
        _lightGlue = lightGlue;
        _sift = sift;
    }

    public IGeometricVerifier Create(bool useLightGlue) => useLightGlue ? _lightGlue : _sift;
}
