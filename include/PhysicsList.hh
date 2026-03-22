#ifndef PHYSICS_LIST_HH
#define PHYSICS_LIST_HH

#include "G4VModularPhysicsList.hh"

class G4GenericMessenger;

class PhysicsList : public G4VModularPhysicsList
{
public:
    PhysicsList();
    ~PhysicsList() override;

    void ConstructParticle() override;
    void ConstructProcess() override;
    void SetCuts() override;

    void SetRangeFactor(G4double val) { fRangeFactor = val; }
    void SetMscStepMax(G4double val)  { fMscStepMax = val; }

private:
    void DefineCommands();

    G4double fRangeFactor;
    G4double fMscStepMax;
    G4GenericMessenger* fMessenger;
};

#endif
