#include "RunAction.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4AnalysisManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4GenericMessenger.hh"

RunAction::RunAction()
    : G4UserRunAction(),
      fOutputFileName("MCSOutput"),
      fMessenger(nullptr)
{
    auto analysisManager = G4AnalysisManager::Instance();
    analysisManager->SetDefaultFileType("root");
    analysisManager->SetVerboseLevel(1);
    analysisManager->SetNtupleMerging(true);

    analysisManager->CreateNtuple("scattering", "MCS scattering data");
    analysisManager->CreateNtupleDColumn("theta_x");
    analysisManager->CreateNtupleDColumn("theta_y");
    analysisManager->CreateNtupleDColumn("theta_space");
    analysisManager->CreateNtupleDColumn("energy_out");
    analysisManager->CreateNtupleDColumn("entry_x");
    analysisManager->CreateNtupleDColumn("entry_y");
    analysisManager->CreateNtupleDColumn("pla_path");
    analysisManager->FinishNtuple();

    DefineCommands();
}

RunAction::~RunAction()
{
    delete fMessenger;
}

void RunAction::BeginOfRunAction(const G4Run* run)
{
    auto analysisManager = G4AnalysisManager::Instance();

    G4String fileName = fOutputFileName;
    analysisManager->OpenFile(fileName);

    G4cout << "Run " << run->GetRunID() << " started" << G4endl;
    G4cout << "Output file: " << fileName << G4endl;
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    G4int nEvents = run->GetNumberOfEvent();
    if (nEvents == 0) return;

    auto analysisManager = G4AnalysisManager::Instance();

    analysisManager->Write();
    analysisManager->CloseFile();

    G4cout << "Run " << run->GetRunID() << " completed: "
           << nEvents << " events" << G4endl;
}

void RunAction::DefineCommands()
{
    fMessenger = new G4GenericMessenger(this, "/MCS/output/",
                                        "Output control");

    fMessenger->DeclareProperty("fileName", fOutputFileName,
        "Output file name (without extension)");
}
