#ifndef DETECTOR_CONSTRUCTION_HH
#define DETECTOR_CONSTRUCTION_HH

#include "G4VUserDetectorConstruction.hh"
#include "G4Material.hh"
#include "G4LogicalVolume.hh"
#include "G4VPhysicalVolume.hh"
#include "G4ThreeVector.hh"
#include "DetectorMessenger.hh"

#include <string>

class G4GenericMessenger;

class DetectorConstruction : public G4VUserDetectorConstruction
{
public:
    DetectorConstruction();
    ~DetectorConstruction() override;

    G4VPhysicalVolume* Construct() override;
    void ConstructSDandField() override;

    enum class GeometryType {
        kSolidPLA,
        kAirOnly,
        kRectilinear,
        kHoneycomb,
        kGyroid,
        kCubic,
        kVoronoi
    };

    void SetGeometryType(const G4String& type);
    void SetInfillPercent(G4double percent);
    void SetCellSize(G4double size);
    void SetWallThickness(G4double thick);
    void SetSampleThickness(G4double thick);
    void SetSampleWidth(G4double width);
    void SetSTLFile(const G4String& filename);
    void SetMaterial(const G4String& name);
    void SetNLayers(G4int n);

    GeometryType GetGeometryType() const { return fGeometryType; }
    G4String GetMaterialName() const { return fMaterialName; }
    G4double GetInfillPercent() const { return fInfillPercent; }
    G4double GetSampleThickness() const { return fSampleThickness; }
    G4LogicalVolume* GetTargetLogical() const { return fLogicTarget; }

private:
    void DefineMaterials();
    G4VPhysicalVolume* ConstructWorld();
    void ConstructSolidPLA();
    void ConstructRectilinearLattice();
    void ConstructStackedRectilinearLattice();
    void ConstructHoneycombLattice();
    void ConstructGyroidLattice();
    void ConstructCubicLattice();
    void ConstructVoronoiLattice();

    void PlaceWallSlab(G4double halfX, G4double halfY, G4double halfZ,
                       G4ThreeVector position, G4RotationMatrix* rot,
                       const G4String& name, G4int copyNo);

    G4Material* fPLA;
    G4Material* fSilicon;
    G4Material* fTungsten;
    G4Material* fTargetMaterial;
    G4Material* fAir;
    G4String fMaterialName;

    GeometryType fGeometryType;
    G4double fInfillPercent;
    G4double fCellSize;
    G4double fWallThickness;
    G4double fSampleThickness;
    G4double fSampleWidth;
    G4String fSTLFile;
    G4int fNLayers;

    G4LogicalVolume* fLogicWorld;
    G4LogicalVolume* fLogicTarget;
    G4LogicalVolume* fLogicPLABlock;

    DetectorMessenger* fMessenger;

    G4bool fConstructed;
};

#endif
