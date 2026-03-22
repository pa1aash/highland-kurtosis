#include "EventAction.hh"

#include "G4Event.hh"
#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"

#include <cmath>

EventAction::EventAction()
    : G4UserEventAction(),
      fEntryPos(0,0,0), fEntryMom(0,0,1), fExitMom(0,0,1),
      fExitEnergy(0), fPLAPathLength(0),
      fHasEntry(false), fHasExit(false)
{
}

void EventAction::BeginOfEventAction(const G4Event*)
{
    fEntryPos = G4ThreeVector(0, 0, 0);
    fEntryMom = G4ThreeVector(0, 0, 1);
    fExitMom  = G4ThreeVector(0, 0, 1);
    fExitEnergy = 0;
    fPLAPathLength = 0;
    fHasEntry = false;
    fHasExit = false;
}

void EventAction::EndOfEventAction(const G4Event*)
{
    if (!fHasExit) return;

    G4double entry_tx = fEntryMom.x() / fEntryMom.z();
    G4double entry_ty = fEntryMom.y() / fEntryMom.z();

    G4double exit_tx = fExitMom.x() / fExitMom.z();
    G4double exit_ty = fExitMom.y() / fExitMom.z();

    G4double theta_x = exit_tx - entry_tx;
    G4double theta_y = exit_ty - entry_ty;
    G4double theta_space = std::sqrt(theta_x*theta_x + theta_y*theta_y);

    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->FillNtupleDColumn(0, 0, theta_x);
    analysisManager->FillNtupleDColumn(0, 1, theta_y);
    analysisManager->FillNtupleDColumn(0, 2, theta_space);
    analysisManager->FillNtupleDColumn(0, 3, fExitEnergy/CLHEP::GeV);
    analysisManager->FillNtupleDColumn(0, 4, fEntryPos.x()/CLHEP::mm);
    analysisManager->FillNtupleDColumn(0, 5, fEntryPos.y()/CLHEP::mm);
    analysisManager->FillNtupleDColumn(0, 6, fPLAPathLength/CLHEP::mm);
    analysisManager->AddNtupleRow(0);
}
