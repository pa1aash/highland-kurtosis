#include "PhysicsList.hh"

#include "G4EmStandardPhysics_option4.hh"
#include "G4EmExtraPhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4HadronElasticPhysics.hh"
#include "G4HadronPhysicsFTFP_BERT.hh"
#include "G4StoppingPhysics.hh"
#include "G4IonPhysics.hh"
#include "G4StepLimiterPhysics.hh"

#include "G4EmParameters.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

PhysicsList::PhysicsList()
    : G4VModularPhysicsList(),
      fRangeFactor(0.04),
      fMscStepMax(0.1*mm),
      fMessenger(nullptr)
{
    SetVerboseLevel(1);

    RegisterPhysics(new G4EmStandardPhysics_option4());

    RegisterPhysics(new G4EmExtraPhysics());
    RegisterPhysics(new G4DecayPhysics());
    RegisterPhysics(new G4HadronElasticPhysics());
    RegisterPhysics(new G4HadronPhysicsFTFP_BERT());
    RegisterPhysics(new G4StoppingPhysics());
    RegisterPhysics(new G4IonPhysics());

    RegisterPhysics(new G4StepLimiterPhysics());

    DefineCommands();
}

PhysicsList::~PhysicsList()
{
    delete fMessenger;
}

void PhysicsList::ConstructParticle()
{
    G4VModularPhysicsList::ConstructParticle();
}

void PhysicsList::ConstructProcess()
{
    G4VModularPhysicsList::ConstructProcess();

    G4EmParameters* param = G4EmParameters::Instance();

    param->SetMscRangeFactor(fRangeFactor);

    param->SetMscSkin(3);

    param->SetMscStepLimitType(fUseSafetyPlus);

    param->SetMscThetaLimit(0.2);

    G4cout << ">>> MCS parameters: RangeFactor=" << fRangeFactor
           << ", StepLimit=UseSafetyPlus, Skin=3" << G4endl;
}

void PhysicsList::SetCuts()
{
    SetDefaultCutValue(1.0*mm);

    G4cout << ">>> Global production cut: 1.0 mm" << G4endl;
    G4cout << ">>> Target region cut: 0.1 mm (set in DetectorConstruction)" << G4endl;
}

void PhysicsList::DefineCommands()
{
    fMessenger = new G4GenericMessenger(this, "/MCS/phys/",
                                        "Physics list control");

    fMessenger->DeclareProperty("rangeFactor", fRangeFactor,
        "MSC range factor (default 0.04)").SetRange("value>0 && value<1");

    fMessenger->DeclarePropertyWithUnit("mscStepMax", "mm", fMscStepMax,
        "Max step size in target (default 0.1 mm)");
}
