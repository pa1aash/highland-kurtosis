#ifndef PRIMARY_GENERATOR_ACTION_HH
#define PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"

class G4GenericMessenger;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    G4double GetBeamEnergy() const;

private:
    void DefineCommands();

    G4ParticleGun* fGun;
    G4double fBeamSigmaXY;
    G4bool   fUsePencilBeam;
    G4GenericMessenger* fMessenger;
};

#endif
