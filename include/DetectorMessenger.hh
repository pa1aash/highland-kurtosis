#ifndef DETECTOR_MESSENGER_HH
#define DETECTOR_MESSENGER_HH

#include "G4UImessenger.hh"
#include "G4UIdirectory.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWithADouble.hh"

class DetectorConstruction;

class DetectorMessenger : public G4UImessenger
{
public:
    DetectorMessenger(DetectorConstruction* det);
    ~DetectorMessenger() override;

    void SetNewValue(G4UIcommand* command, G4String newValue) override;

private:
    DetectorConstruction* fDetector;

    G4UIdirectory*             fDetDir;
    G4UIcmdWithAString*        fGeomTypeCmd;
    G4UIcmdWithADouble*        fInfillCmd;
    G4UIcmdWithADoubleAndUnit* fCellSizeCmd;
    G4UIcmdWithADoubleAndUnit* fWallThickCmd;
    G4UIcmdWithADoubleAndUnit* fSampleThickCmd;
    G4UIcmdWithADoubleAndUnit* fSampleWidthCmd;
    G4UIcmdWithAString*        fSTLFileCmd;
};

#endif
