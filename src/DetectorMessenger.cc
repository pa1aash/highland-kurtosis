#include "DetectorMessenger.hh"
#include "DetectorConstruction.hh"
#include "G4SystemOfUnits.hh"

DetectorMessenger::DetectorMessenger(DetectorConstruction* det)
    : G4UImessenger(), fDetector(det)
{
    fDetDir = new G4UIdirectory("/MCS/det/");
    fDetDir->SetGuidance("Detector geometry control.");

    fGeomTypeCmd = new G4UIcmdWithAString("/MCS/det/geometry", this);
    fGeomTypeCmd->SetGuidance("Geometry type: solid air rectilinear honeycomb gyroid cubic voronoi");
    fGeomTypeCmd->SetParameterName("type", false);
    fGeomTypeCmd->SetCandidates("solid air rectilinear honeycomb gyroid cubic voronoi");
    fGeomTypeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fInfillCmd = new G4UIcmdWithADouble("/MCS/det/infill", this);
    fInfillCmd->SetGuidance("Infill percentage (0-100).");
    fInfillCmd->SetParameterName("percent", false);
    fInfillCmd->SetRange("percent>=0. && percent<=100.");
    fInfillCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fCellSizeCmd = new G4UIcmdWithADoubleAndUnit("/MCS/det/cellSize", this);
    fCellSizeCmd->SetGuidance("Lattice cell period.");
    fCellSizeCmd->SetParameterName("size", false);
    fCellSizeCmd->SetDefaultUnit("mm");
    fCellSizeCmd->SetUnitCategory("Length");
    fCellSizeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fWallThickCmd = new G4UIcmdWithADoubleAndUnit("/MCS/det/wallThickness", this);
    fWallThickCmd->SetGuidance("PLA wall thickness.");
    fWallThickCmd->SetParameterName("thick", false);
    fWallThickCmd->SetDefaultUnit("mm");
    fWallThickCmd->SetUnitCategory("Length");
    fWallThickCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSampleThickCmd = new G4UIcmdWithADoubleAndUnit("/MCS/det/sampleThickness", this);
    fSampleThickCmd->SetGuidance("Sample thickness along beam (z).");
    fSampleThickCmd->SetParameterName("thick", false);
    fSampleThickCmd->SetDefaultUnit("mm");
    fSampleThickCmd->SetUnitCategory("Length");
    fSampleThickCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSampleWidthCmd = new G4UIcmdWithADoubleAndUnit("/MCS/det/sampleWidth", this);
    fSampleWidthCmd->SetGuidance("Sample transverse width.");
    fSampleWidthCmd->SetParameterName("width", false);
    fSampleWidthCmd->SetDefaultUnit("mm");
    fSampleWidthCmd->SetUnitCategory("Length");
    fSampleWidthCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fSTLFileCmd = new G4UIcmdWithAString("/MCS/det/stlFile", this);
    fSTLFileCmd->SetGuidance("Path to STL file for gyroid/voronoi geometry.");
    fSTLFileCmd->SetParameterName("filename", false);
    fSTLFileCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fMaterialCmd = new G4UIcmdWithAString("/MCS/det/material", this);
    fMaterialCmd->SetGuidance("Target material: PLA silicon tungsten");
    fMaterialCmd->SetParameterName("material", false);
    fMaterialCmd->SetCandidates("PLA silicon tungsten");
    fMaterialCmd->AvailableForStates(G4State_PreInit, G4State_Idle);

    fNLayersCmd = new G4UIcmdWithAnInteger("/MCS/det/nLayers", this);
    fNLayersCmd->SetGuidance("Number of independent rectilinear layers stacked along z (1-100).");
    fNLayersCmd->SetParameterName("nLayers", false);
    fNLayersCmd->SetDefaultValue(1);
    fNLayersCmd->SetRange("nLayers>=1 && nLayers<=100");
    fNLayersCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

DetectorMessenger::~DetectorMessenger()
{
    delete fGeomTypeCmd;
    delete fInfillCmd;
    delete fCellSizeCmd;
    delete fWallThickCmd;
    delete fSampleThickCmd;
    delete fSampleWidthCmd;
    delete fSTLFileCmd;
    delete fMaterialCmd;
    delete fNLayersCmd;
    delete fDetDir;
}

void DetectorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if (command == fGeomTypeCmd) {
        fDetector->SetGeometryType(newValue);
    }
    else if (command == fInfillCmd) {
        fDetector->SetInfillPercent(fInfillCmd->GetNewDoubleValue(newValue));
    }
    else if (command == fCellSizeCmd) {
        fDetector->SetCellSize(fCellSizeCmd->GetNewDoubleValue(newValue));
    }
    else if (command == fWallThickCmd) {
        fDetector->SetWallThickness(fWallThickCmd->GetNewDoubleValue(newValue));
    }
    else if (command == fSampleThickCmd) {
        fDetector->SetSampleThickness(fSampleThickCmd->GetNewDoubleValue(newValue));
    }
    else if (command == fSampleWidthCmd) {
        fDetector->SetSampleWidth(fSampleWidthCmd->GetNewDoubleValue(newValue));
    }
    else if (command == fSTLFileCmd) {
        fDetector->SetSTLFile(newValue);
    }
    else if (command == fMaterialCmd) {
        fDetector->SetMaterial(newValue);
    }
    else if (command == fNLayersCmd) {
        fDetector->SetNLayers(fNLayersCmd->GetNewIntValue(newValue));
    }
}
