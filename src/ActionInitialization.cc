#include "ActionInitialization.hh"
#include "PrimaryGeneratorAction.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "SteppingAction.hh"

#include "G4GenericMessenger.hh"
#include "G4SystemOfUnits.hh"

ActionInitialization::ActionInitialization()
    : G4VUserActionInitialization(),
      fMessenger(nullptr),
      fParticleName("e-"),
      fBeamSigmaXY(5.0*mm),
      fUsePencilBeam(false)
{
    DefineCommands();
}

ActionInitialization::~ActionInitialization()
{
    delete fMessenger;
}

void ActionInitialization::BuildForMaster() const
{
    SetUserAction(new RunAction());
}

void ActionInitialization::Build() const
{
    auto* gun = new PrimaryGeneratorAction(fParticleName, fBeamSigmaXY,
                                           fUsePencilBeam);
    SetUserAction(gun);
    SetUserAction(new RunAction());

    auto* eventAction = new EventAction();
    SetUserAction(eventAction);
    SetUserAction(new SteppingAction(eventAction));
}

void ActionInitialization::DefineCommands()
{
    fMessenger = new G4GenericMessenger(this, "/MCS/gun/",
                                        "Beam control");

    fMessenger->DeclareProperty("pencilBeam", fUsePencilBeam,
        "Use pencil beam (true) or Gaussian spot (false)");

    fMessenger->DeclarePropertyWithUnit("beamSigma", "mm", fBeamSigmaXY,
        "Gaussian beam spot sigma (default 5 mm)");

    fMessenger->DeclareProperty("particle", fParticleName,
        "Primary particle type (default e-)");
}
