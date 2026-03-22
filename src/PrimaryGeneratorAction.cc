#include "PrimaryGeneratorAction.hh"

#include "G4ParticleTable.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"
#include "Randomize.hh"

PrimaryGeneratorAction::PrimaryGeneratorAction()
    : G4VUserPrimaryGeneratorAction(),
      fGun(nullptr),
      fBeamSigmaXY(5.0*mm),
      fUsePencilBeam(false),
      fMessenger(nullptr)
{
    fGun = new G4ParticleGun(1);

    G4ParticleDefinition* electron =
        G4ParticleTable::GetParticleTable()->FindParticle("e-");
    fGun->SetParticleDefinition(electron);

    fGun->SetParticleEnergy(4.0*GeV);
    fGun->SetParticleMomentumDirection(G4ThreeVector(0, 0, 1));
    fGun->SetParticlePosition(G4ThreeVector(0, 0, -50*mm));

    DefineCommands();
}

PrimaryGeneratorAction::~PrimaryGeneratorAction()
{
    delete fGun;
    delete fMessenger;
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

void PrimaryGeneratorAction::DefineCommands()
{
    fMessenger = new G4GenericMessenger(this, "/MCS/gun/",
                                        "Beam control");

    fMessenger->DeclareProperty("pencilBeam", fUsePencilBeam,
        "Use pencil beam (true) or Gaussian spot (false)");

    fMessenger->DeclarePropertyWithUnit("beamSigma", "mm", fBeamSigmaXY,
        "Gaussian beam spot sigma (default 5 mm)");
}
