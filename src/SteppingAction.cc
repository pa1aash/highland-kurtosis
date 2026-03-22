#include "SteppingAction.hh"
#include "EventAction.hh"

#include "G4Step.hh"
#include "G4Track.hh"
#include "G4StepPoint.hh"
#include "G4VPhysicalVolume.hh"
#include "G4SystemOfUnits.hh"

SteppingAction::SteppingAction(EventAction* eventAction)
    : G4UserSteppingAction(),
      fEventAction(eventAction)
{
}

void SteppingAction::UserSteppingAction(const G4Step* step)
{
    if (step->GetTrack()->GetTrackID() != 1) return;

    G4StepPoint* prePoint  = step->GetPreStepPoint();
    G4StepPoint* postPoint = step->GetPostStepPoint();

    G4VPhysicalVolume* preVol  = prePoint->GetPhysicalVolume();
    G4VPhysicalVolume* postVol = postPoint->GetPhysicalVolume();

    if (!preVol || !postVol) return;

    G4String preName  = preVol->GetName();
    G4String postName = postVol->GetName();

    auto isTargetVolume = [](const G4String& name) -> bool {
        return (name == "Target" || name == "SolidPLA" || name == "PLAFill" ||
                name == "XWall"  || name == "YWall"    || name == "HCWall"  ||
                name == "GVoxPLA"|| name == "VVoxPLA"  || name == "Gyroid"  ||
                name == "Voronoi"|| name == "CubZWall" || name == "CubXWall"||
                name == "CubYWall");
    };

    auto isPLAVolume = [](const G4String& name) -> bool {
        return (name == "SolidPLA" || name == "PLAFill" ||
                name == "XWall"  || name == "YWall"    || name == "HCWall"  ||
                name == "GVoxPLA"|| name == "VVoxPLA"  || name == "Gyroid"  ||
                name == "Voronoi"|| name == "CubZWall" || name == "CubXWall"||
                name == "CubYWall");
    };

    G4bool preInTarget  = isTargetVolume(preName);
    G4bool postInTarget = isTargetVolume(postName);

    if (isPLAVolume(preName)) {
        fEventAction->AddPLAPathLength(step->GetStepLength());
    }

    if (!fEventAction->HasEntry() && !isTargetVolume(preName) && postInTarget) {
        fEventAction->SetEntryPosition(postPoint->GetPosition());
        fEventAction->SetEntryMomentum(postPoint->GetMomentumDirection());
    }

    if (preInTarget && !postInTarget && postName == "World") {
        fEventAction->SetExitMomentum(postPoint->GetMomentumDirection());
        fEventAction->SetExitEnergy(postPoint->GetKineticEnergy());
    }
}
