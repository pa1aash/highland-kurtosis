#ifndef PRIMARY_GENERATOR_ACTION_HH
#define PRIMARY_GENERATOR_ACTION_HH

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"
#include "G4String.hh"

class G4GenericMessenger;

class PrimaryGeneratorAction : public G4VUserPrimaryGeneratorAction
{
public:
    PrimaryGeneratorAction();
    ~PrimaryGeneratorAction() override;

    void GeneratePrimaries(G4Event* event) override;

    G4double GetBeamEnergy() const;
    void SetParticle(const G4String& name);
    G4String GetParticleName() const { return fParticleName; }

private:
    void DefineCommands();

    G4ParticleGun* fGun;
    G4double fBeamSigmaXY;
    G4bool   fUsePencilBeam;
    G4String fParticleName;
    G4GenericMessenger* fMessenger;
};

#endif
