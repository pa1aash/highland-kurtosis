#ifndef EVENT_ACTION_HH
#define EVENT_ACTION_HH

#include "G4UserEventAction.hh"
#include "G4ThreeVector.hh"

class EventAction : public G4UserEventAction
{
public:
    EventAction();
    ~EventAction() override = default;

    void BeginOfEventAction(const G4Event* event) override;
    void EndOfEventAction(const G4Event* event) override;

    void SetEntryPosition(G4ThreeVector pos) { fEntryPos = pos; fHasEntry = true; }
    void SetEntryMomentum(G4ThreeVector mom) { fEntryMom = mom; }
    void SetExitMomentum(G4ThreeVector mom)  { fExitMom = mom; fHasExit = true; }
    void SetExitEnergy(G4double e)           { fExitEnergy = e; }
    void AddPLAPathLength(G4double len)      { fPLAPathLength += len; }

    G4bool HasEntry() const { return fHasEntry; }
    G4bool HasExit() const  { return fHasExit; }

private:
    G4ThreeVector fEntryPos;
    G4ThreeVector fEntryMom;
    G4ThreeVector fExitMom;
    G4double fExitEnergy;
    G4double fPLAPathLength;
    G4bool fHasEntry;
    G4bool fHasExit;
};

#endif
