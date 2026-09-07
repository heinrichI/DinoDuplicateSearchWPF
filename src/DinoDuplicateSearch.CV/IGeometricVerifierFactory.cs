namespace DinoDuplicateSearch.CV;

public interface IGeometricVerifierFactory
{
    IGeometricVerifier Create(bool useLightGlue);
}
