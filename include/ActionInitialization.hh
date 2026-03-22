#ifndef ACTION_INITIALIZATION_HH
#define ACTION_INITIALIZATION_HH

#include "G4VUserActionInitialization.hh"
#include "G4String.hh"
#include "G4Types.hh"

class G4GenericMessenger;

class ActionInitialization : public G4VUserActionInitialization
{
public:
    ActionInitialization();
    ~ActionInitialization() override;

    void BuildForMaster() const override;
    void Build() const override;

private:
    void DefineCommands();

    G4GenericMessenger* fMessenger;

    // Pre-init beam parameters (set via /MCS/gun/ before /run/initialize)
    G4String fParticleName;
    G4double fBeamSigmaXY;
    G4bool   fUsePencilBeam;
};

#endif
