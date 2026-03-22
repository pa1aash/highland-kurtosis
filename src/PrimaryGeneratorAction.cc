#include "PrimaryGeneratorAction.hh"

#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction(const G4String& particleName,
                                               G4double beamSigma,
                                               G4bool pencilBeam)
    : G4VUserPrimaryGeneratorAction(),
      fGun(nullptr),
      fBeamSigmaXY(beamSigma),
      fUsePencilBeam(pencilBeam),
      fParticleName(particleName)
{
    fGun = new G4ParticleGun(1);

    // Apply particle from pre-init configuration
    G4ParticleDefinition* particle =
        G4ParticleTable::GetParticleTable()->FindParticle(fParticleName);
    if (particle) {
        fGun->SetParticleDefinition(particle);
    } else {
        G4cerr << "*** PrimaryGeneratorAction: particle \""
               << fParticleName << "\" not found, defaulting to e-" << G4endl;
        fGun->SetParticleDefinition(
            G4ParticleTable::GetParticleTable()->FindParticle("e-"));
        fParticleName = "e-";
    }

    fGun->SetParticleEnergy(4.0*GeV);
    fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    fGun->SetParticlePosition(G4ThreeVector(0, 0, -50*mm));
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fGun;
}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event)
{
    if (fUsePencilBeam) {
        fGun->SetParticlePosition(G4ThreeVector(0, 0, -50*mm));
    } else {
        G4double x = G4RandGauss::shoot(0, fBeamSigmaXY);
        G4double y = G4RandGauss::shoot(0, fBeamSigmaXY);
        fGun->SetParticlePosition(G4ThreeVector(x, y, -50*mm));
    }

    fGun->GeneratePrimaryVertex(event);
}

G4double PrimaryGeneratorAction::GetBeamEnergy() const
{
    return fGun->GetParticleEnergy();
}

void PrimaryGeneratorAction::SetParticle(const G4String& name)
{
    G4ParticleDefinition* particle =
        G4ParticleTable::GetParticleTable()->FindParticle(name);
    if (particle) {
        fGun->SetParticleDefinition(particle);
        fParticleName = name;
    } else {
        G4cerr << "*** PrimaryGeneratorAction::SetParticle: particle \""
               << name << "\" not found in G4ParticleTable." << G4endl;
    }
}
