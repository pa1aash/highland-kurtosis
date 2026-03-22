#include "DetectorConstruction.hh"
#include "DetectorMessenger.hh"

#include "G4Box.hh"
#include "G4Colour.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4PhysicalConstants.hh"
#include "G4ProductionCuts.hh"
#include "G4Region.hh"
#include "G4RotationMatrix.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "Randomize.hh"

#ifdef USE_CADMESH
#include "CADMesh.hh"
#endif

#include <algorithm>
#include <cmath>
#include <vector>

DetectorConstruction::DetectorConstruction()
    : G4VUserDetectorConstruction(), fPLA(nullptr), fSilicon(nullptr),
      fTungsten(nullptr), fTargetMaterial(nullptr), fAir(nullptr),
      fMaterialName("PLA"),
      fGeometryType(GeometryType::kSolidPLA), fInfillPercent(100.0),
      fCellSize(4.0 * mm), fWallThickness(0.4 * mm),
      fSampleThickness(10.0 * mm), fSampleWidth(20.0 * mm), fSTLFile(""),
      fLogicWorld(nullptr), fLogicTarget(nullptr), fLogicPLABlock(nullptr),
      fConstructed(false) {
  fMessenger = new DetectorMessenger(this);
  DefineMaterials();
}

DetectorConstruction::~DetectorConstruction() { delete fMessenger; }

void DetectorConstruction::DefineMaterials() {
  G4NistManager *nist = G4NistManager::Instance();
  fAir = nist->FindOrBuildMaterial("G4_AIR");

  fPLA = new G4Material("PLA", 1.24 * g / cm3, 3);
  fPLA->AddElement(nist->FindOrBuildElement("C"), 3);
  fPLA->AddElement(nist->FindOrBuildElement("H"), 4);
  fPLA->AddElement(nist->FindOrBuildElement("O"), 2);

  fSilicon = nist->FindOrBuildMaterial("G4_Si");
  fTungsten = nist->FindOrBuildMaterial("G4_W");

  fTargetMaterial = fPLA;

  G4cout << "PLA X0 = " << fPLA->GetRadlen() / cm << " cm ("
         << fPLA->GetRadlen() / mm << " mm)" << G4endl;
  G4cout << "PLA density = " << fPLA->GetDensity() / (g / cm3) << " g/cm3"
         << G4endl;
  G4cout << "Si  X0 = " << fSilicon->GetRadlen() / mm << " mm" << G4endl;
  G4cout << "W   X0 = " << fTungsten->GetRadlen() / mm << " mm" << G4endl;
  G4cout << "Air X0 = " << fAir->GetRadlen() / m << " m" << G4endl;
}

void DetectorConstruction::SetGeometryType(const G4String &type) {
  if (type == "solid")
    fGeometryType = GeometryType::kSolidPLA;
  else if (type == "air")
    fGeometryType = GeometryType::kAirOnly;
  else if (type == "rectilinear")
    fGeometryType = GeometryType::kRectilinear;
  else if (type == "honeycomb")
    fGeometryType = GeometryType::kHoneycomb;
  else if (type == "gyroid")
    fGeometryType = GeometryType::kGyroid;
  else if (type == "cubic")
    fGeometryType = GeometryType::kCubic;
  else if (type == "voronoi")
    fGeometryType = GeometryType::kVoronoi;
  else {
    G4cerr << "ERROR: Unknown geometry type '" << type << "'" << G4endl;
    G4cerr << "Valid: solid air rectilinear honeycomb gyroid cubic voronoi"
           << G4endl;
    return;
  }
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetInfillPercent(G4double p) {
  fInfillPercent = p;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetCellSize(G4double s) {
  fCellSize = s;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetWallThickness(G4double t) {
  fWallThickness = t;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetSampleThickness(G4double t) {
  fSampleThickness = t;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetSampleWidth(G4double w) {
  fSampleWidth = w;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetSTLFile(const G4String &f) {
  fSTLFile = f;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

void DetectorConstruction::SetMaterial(const G4String &name) {
  if (name == "PLA") {
    fTargetMaterial = fPLA;
  } else if (name == "silicon") {
    fTargetMaterial = fSilicon;
  } else if (name == "tungsten") {
    fTargetMaterial = fTungsten;
  } else {
    G4cerr << "ERROR: Unknown material '" << name << "'" << G4endl;
    G4cerr << "Valid: PLA silicon tungsten" << G4endl;
    return;
  }
  fMaterialName = name;
  G4cout << ">>> Target material set to " << name
         << " (X0=" << fTargetMaterial->GetRadlen() / mm << " mm)" << G4endl;
  if (fConstructed)
    G4RunManager::GetRunManager()->ReinitializeGeometry();
}

G4VPhysicalVolume *DetectorConstruction::Construct() {
  G4VPhysicalVolume *physWorld = ConstructWorld();

  switch (fGeometryType) {
  case GeometryType::kSolidPLA:
    ConstructSolidPLA();
    break;
  case GeometryType::kAirOnly:
    G4cout << ">>> Air-only (no target)" << G4endl;
    break;
  case GeometryType::kRectilinear:
    ConstructRectilinearLattice();
    break;
  case GeometryType::kHoneycomb:
    ConstructHoneycombLattice();
    break;
  case GeometryType::kGyroid:
    ConstructGyroidLattice();
    break;
  case GeometryType::kCubic:
    ConstructCubicLattice();
    break;
  case GeometryType::kVoronoi:
    ConstructVoronoiLattice();
    break;
  }

  fConstructed = true;
  return physWorld;
}

G4VPhysicalVolume *DetectorConstruction::ConstructWorld() {
  G4double worldHalf = 100.0 * mm;
  G4Box *solidWorld = new G4Box("World", worldHalf, worldHalf, worldHalf);
  fLogicWorld = new G4LogicalVolume(solidWorld, fAir, "World");
  fLogicWorld->SetVisAttributes(G4VisAttributes::GetInvisible());

  return new G4PVPlacement(nullptr, G4ThreeVector(), fLogicWorld, "World",
                           nullptr, false, 0, true);
}

static void MakeTargetRegion(G4LogicalVolume *lv) {
  G4Region *reg = new G4Region("Target");
  reg->AddRootLogicalVolume(lv);
  G4ProductionCuts *cuts = new G4ProductionCuts();
  cuts->SetProductionCut(0.1 * mm);
  reg->SetProductionCuts(cuts);
}

void DetectorConstruction::PlaceWallSlab(G4double halfX, G4double halfY,
                                         G4double halfZ, G4ThreeVector position,
                                         G4RotationMatrix *rot,
                                         const G4String &name, G4int copyNo) {
  G4Box *solid = new G4Box(name, halfX, halfY, halfZ);
  G4LogicalVolume *logic = new G4LogicalVolume(solid, fTargetMaterial, name);
  logic->SetUserLimits(new G4UserLimits(0.1 * mm));
  auto *vis = new G4VisAttributes(G4Colour(0.8, 0.6, 0.2, 0.5));
  vis->SetForceSolid(true);
  logic->SetVisAttributes(vis);

  new G4PVPlacement(rot, position, logic, name, fLogicTarget, false, copyNo,
                    false);
}

void DetectorConstruction::ConstructSolidPLA() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solid = new G4Box("SolidPLA", halfW, halfW, halfT);
  fLogicPLABlock = new G4LogicalVolume(solid, fTargetMaterial, "SolidPLA");
  fLogicPLABlock->SetUserLimits(new G4UserLimits(0.1 * mm));

  auto *vis = new G4VisAttributes(G4Colour(0.8, 0.6, 0.2, 0.7));
  vis->SetForceSolid(true);
  fLogicPLABlock->SetVisAttributes(vis);

  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicPLABlock, "SolidPLA",
                    fLogicWorld, false, 0, true);

  fLogicTarget = fLogicPLABlock;
  MakeTargetRegion(fLogicPLABlock);

  G4cout << ">>> Solid PLA: " << fSampleWidth / mm << " x " << fSampleWidth / mm
         << " x " << fSampleThickness / mm << " mm^3"
         << ", x/X0 = " << fSampleThickness / fTargetMaterial->GetRadlen()
         << " (" << fMaterialName << ")" << G4endl;
}

void DetectorConstruction::ConstructRectilinearLattice() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solidTarget = new G4Box("Target", halfW, halfW, halfT);
  fLogicTarget = new G4LogicalVolume(solidTarget, fAir, "Target");
  fLogicTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicTarget, "Target",
                    fLogicWorld, false, 0, true);
  MakeTargetRegion(fLogicTarget);

  if (fInfillPercent >= 100.0) {
    G4Box *sf = new G4Box("PLAFill", halfW, halfW, halfT);
    G4LogicalVolume *lf = new G4LogicalVolume(sf, fTargetMaterial, "PLAFill");
    lf->SetUserLimits(new G4UserLimits(0.1 * mm));
    new G4PVPlacement(nullptr, G4ThreeVector(), lf, "PLAFill", fLogicTarget,
                      false, 0, true);
    return;
  }

  G4double halfWallT = fWallThickness / 2.0;
  G4int nCells = std::max(1, static_cast<G4int>(fSampleWidth / fCellSize));
  G4int copyNo = 0;

  for (G4int i = 0; i <= nCells; i++) {
    G4double xPos = -halfW + i * fCellSize;
    xPos = std::clamp(xPos, -halfW + halfWallT, halfW - halfWallT);
    PlaceWallSlab(halfWallT, halfW, halfT, G4ThreeVector(xPos, 0, 0), nullptr,
                  "XWall", copyNo++);
  }

  for (G4int j = 0; j <= nCells; j++) {
    G4double yPos = -halfW + j * fCellSize;
    yPos = std::clamp(yPos, -halfW + halfWallT, halfW - halfWallT);
    PlaceWallSlab(halfW, halfWallT, halfT, G4ThreeVector(0, yPos, 0), nullptr,
                  "YWall", copyNo++);
  }

  G4cout << ">>> Rectilinear: " << nCells << "x" << nCells << " cells, "
         << "cell=" << fCellSize / mm << " mm, wall=" << fWallThickness / mm
         << " mm, target infill=" << fInfillPercent << "%" << G4endl;
}

void DetectorConstruction::ConstructHoneycombLattice() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solidTarget = new G4Box("Target", halfW, halfW, halfT);
  fLogicTarget = new G4LogicalVolume(solidTarget, fAir, "Target");
  fLogicTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicTarget, "Target",
                    fLogicWorld, false, 0, true);
  MakeTargetRegion(fLogicTarget);

  if (fInfillPercent >= 100.0) {
    G4Box *sf = new G4Box("PLAFill", halfW, halfW, halfT);
    G4LogicalVolume *lf = new G4LogicalVolume(sf, fTargetMaterial, "PLAFill");
    lf->SetUserLimits(new G4UserLimits(0.1 * mm));
    new G4PVPlacement(nullptr, G4ThreeVector(), lf, "PLAFill", fLogicTarget,
                      false, 0, true);
    return;
  }

  G4double targetFrac = fInfillPercent / 100.0;
  G4double wallT = fWallThickness;
  G4double halfWallT = wallT / 2.0;
  G4double sW = fSampleWidth;
  G4double beamSigma = 5.0 * mm;

  G4double cos60 = 0.5, sin60 = std::sqrt(3.0) / 2.0;
  G4double cos120 = -0.5, sin120 = sin60;

  auto hexInfill = [&](G4double dd) -> G4double {
    G4double aa = dd / std::sqrt(3.0);
    G4double hA = aa / 2.0;
    G4double colSp = 1.5 * aa;
    G4double rowSp = dd;
    G4int nIn = 0, nProbe = 50000;
    for (G4int p = 0; p < nProbe; p++) {
      G4double px = G4RandGauss::shoot(0.0, beamSigma);
      G4double py = G4RandGauss::shoot(0.0, beamSigma);
      G4int cNear = static_cast<G4int>(std::round(px / colSp));
      G4int rNear = static_cast<G4int>(std::round(py / rowSp));
      bool hit = false;
      for (G4int dc = -2; dc <= 2 && !hit; dc++) {
        G4int c = cNear + dc;
        G4double cx = c * colSp;
        G4double yOff = (std::abs(c) % 2) ? rowSp * 0.5 : 0;
        for (G4int dr = -2; dr <= 2 && !hit; dr++) {
          G4double cy = (rNear + dr) * rowSp + yOff;
          {
            G4double dx = px - cx;
            G4double dy = py - (cy + dd * 0.5);
            if (std::abs(dx) < hA && std::abs(dy) < halfWallT) {
              hit = true;
              break;
            }
          }
          {
            G4double dx = px - (cx + 0.75 * aa);
            G4double dy = py - (cy + dd * 0.25);
            G4double lx = cos120 * dx + sin120 * dy;
            G4double ly = -sin120 * dx + cos120 * dy;
            if (std::abs(lx) < hA && std::abs(ly) < halfWallT) {
              hit = true;
              break;
            }
          }
          {
            G4double dx = px - (cx + 0.75 * aa);
            G4double dy = py - (cy - dd * 0.25);
            G4double lx = cos60 * dx + sin60 * dy;
            G4double ly = -sin60 * dx + cos60 * dy;
            if (std::abs(lx) < hA && std::abs(ly) < halfWallT) {
              hit = true;
              break;
            }
          }
        }
      }
      if (hit)
        nIn++;
    }
    return static_cast<G4double>(nIn) / nProbe;
  };

G4double dLo = wallT * 1.5;
G4double dHi = sW / 2.0;
G4double d = fCellSize;
for (G4int iter = 0; iter < 25; iter++) {
  G4double frac = hexInfill(d);
  if (frac < targetFrac)
    dHi = d;
  else
    dLo = d;
  d = (dLo + dHi) / 2.0;
}

G4cout << ">>> Honeycomb bisection: d=" << d / mm << " mm for "
       << fInfillPercent << "% infill" << G4endl;

G4double a = d / std::sqrt(3.0);
G4double halfA = a / 2.0;
G4double colSp = 1.5 * a;
G4double rowSp = d;

G4double pixelSize = 0.1 * mm;
G4int nPx = static_cast<G4int>(fSampleWidth / pixelSize);
G4int nPy = nPx;
G4double halfPx = pixelSize / 2.0;

G4Box *solidVox = new G4Box("HCVox", halfPx, halfPx, halfT);
G4LogicalVolume *logicVox = new G4LogicalVolume(solidVox, fTargetMaterial, "HCVoxPLA");
logicVox->SetUserLimits(new G4UserLimits(0.5 * mm));
auto *vis = new G4VisAttributes(G4Colour(0.8, 0.6, 0.2, 0.5));
vis->SetForceSolid(true);
logicVox->SetVisAttributes(vis);

G4int nPlaced = 0;
for (G4int ix = 0; ix < nPx; ix++) {
  G4double x = -halfW + (ix + 0.5) * pixelSize;
  for (G4int iy = 0; iy < nPy; iy++) {
    G4double y = -halfW + (iy + 0.5) * pixelSize;

    G4int cNear = static_cast<G4int>(std::round(x / colSp));
    G4int rNear = static_cast<G4int>(std::round(y / rowSp));
    bool hit = false;
    for (G4int dc = -2; dc <= 2 && !hit; dc++) {
      G4int c = cNear + dc;
      G4double cx = c * colSp;
      G4double yOff = (std::abs(c) % 2) ? rowSp * 0.5 : 0;
      for (G4int dr = -2; dr <= 2 && !hit; dr++) {
        G4double cy = (rNear + dr) * rowSp + yOff;
        {
          G4double dx = x - cx;
          G4double dy = y - (cy + d * 0.5);
          if (std::abs(dx) < halfA && std::abs(dy) < halfWallT) {
            hit = true;
            break;
          }
        }
        {
          G4double dx = x - (cx + 0.75 * a);
          G4double dy = y - (cy + d * 0.25);
          G4double lx = cos120 * dx + sin120 * dy;
          G4double ly = -sin120 * dx + cos120 * dy;
          if (std::abs(lx) < halfA && std::abs(ly) < halfWallT) {
            hit = true;
            break;
          }
        }
        {
          G4double dx = x - (cx + 0.75 * a);
          G4double dy = y - (cy - d * 0.25);
          G4double lx = cos60 * dx + sin60 * dy;
          G4double ly = -sin60 * dx + cos60 * dy;
          if (std::abs(lx) < halfA && std::abs(ly) < halfWallT) {
            hit = true;
            break;
          }
        }
      }
    }
    if (hit) {
      new G4PVPlacement(nullptr, G4ThreeVector(x, y, 0), logicVox, "HCWall",
                        fLogicTarget, false, nPlaced, false);
      nPlaced++;
    }
  }
}

G4cout << ">>> Honeycomb (voxelised): " << nPlaced << "/" << (nPx * nPy)
       << " columns (" << 100.0 * nPlaced / (nPx * nPy) << "% infill), "
       << "d=" << d / mm << " mm, pixel=" << pixelSize / mm << " mm" << G4endl;
}

void DetectorConstruction::ConstructGyroidLattice() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solidTarget = new G4Box("Target", halfW, halfW, halfT);
  fLogicTarget = new G4LogicalVolume(solidTarget, fAir, "Target");
  fLogicTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicTarget, "Target",
                    fLogicWorld, false, 0, true);
  MakeTargetRegion(fLogicTarget);

  if (fInfillPercent >= 100.0) {
    G4Box *sf = new G4Box("PLAFill", halfW, halfW, halfT);
    G4LogicalVolume *lf = new G4LogicalVolume(sf, fTargetMaterial, "PLAFill");
    lf->SetUserLimits(new G4UserLimits(0.1 * mm));
    new G4PVPlacement(nullptr, G4ThreeVector(), lf, "PLAFill", fLogicTarget,
                      false, 0, true);
    return;
  }

#ifdef USE_CADMESH
  if (!fSTLFile.empty()) {
    G4cout << ">>> Loading gyroid STL: " << fSTLFile << G4endl;
    auto mesh = CADMesh::TessellatedMesh::FromSTL(fSTLFile);
    mesh->SetScale(mm);
    mesh->SetOffset(G4ThreeVector(-halfW, -halfW, -halfT));
    G4VSolid *solidG = mesh->GetSolid();
    G4LogicalVolume *logicG = new G4LogicalVolume(solidG, fTargetMaterial, "Gyroid");
    logicG->SetUserLimits(new G4UserLimits(0.1 * mm));
    auto *vis = new G4VisAttributes(G4Colour(0.2, 0.7, 0.3, 0.5));
    vis->SetForceSolid(true);
    logicG->SetVisAttributes(vis);
    new G4PVPlacement(nullptr, G4ThreeVector(), logicG, "Gyroid", fLogicTarget,
                      false, 0, true);
    return;
  }
#endif

  G4cout << ">>> Building voxelised gyroid..." << G4endl;

  G4double voxelSize = 0.4 * mm;
  G4int nX = static_cast<G4int>(fSampleWidth / voxelSize);
  G4int nY = nX;
  G4int nZ = static_cast<G4int>(fSampleThickness / voxelSize);

  while (static_cast<long long>(nX) * nY * nZ > 500000) {
    voxelSize *= 1.2;
    nX = static_cast<G4int>(fSampleWidth / voxelSize);
    nY = nX;
    nZ = static_cast<G4int>(fSampleThickness / voxelSize);
  }

  G4double k = 2.0 * CLHEP::pi / fCellSize;

  G4double targetFrac = fInfillPercent / 100.0;
  G4double tLo = 0.0, tHi = 1.5, threshold = 0.75;
  G4int nSamp = 50000;
  for (G4int iter = 0; iter < 25; iter++) {
    G4int nIn = 0;
    for (G4int s = 0; s < nSamp; s++) {
      G4double sx = G4UniformRand() * fSampleWidth - halfW;
      G4double sy = G4UniformRand() * fSampleWidth - halfW;
      G4double sz = G4UniformRand() * fSampleThickness - halfT;
      G4double F = std::sin(k * sx) * std::cos(k * sy) +
                   std::sin(k * sy) * std::cos(k * sz) +
                   std::sin(k * sz) * std::cos(k * sx);
      if (std::abs(F) < threshold)
        nIn++;
    }
    G4double frac = static_cast<G4double>(nIn) / nSamp;
    if (frac < targetFrac)
      tLo = threshold;
    else
      tHi = threshold;
    threshold = (tLo + tHi) / 2.0;
  }

  G4cout << ">>> Gyroid threshold=" << threshold << " for " << fInfillPercent
         << "% infill" << G4endl;

  G4double halfV = voxelSize / 2.0;
  G4Box *solidVox = new G4Box("GVox", halfV, halfV, halfV);
  G4LogicalVolume *logicVox = new G4LogicalVolume(solidVox, fTargetMaterial, "GVoxPLA");
  logicVox->SetUserLimits(new G4UserLimits(voxelSize * 0.5));
  auto *vis = new G4VisAttributes(G4Colour(0.2, 0.7, 0.3, 0.3));
  vis->SetForceSolid(true);
  logicVox->SetVisAttributes(vis);

  G4int nPlaced = 0, nTotal = 0;
  for (G4int ix = 0; ix < nX; ix++) {
    G4double x = -halfW + (ix + 0.5) * voxelSize;
    for (G4int iy = 0; iy < nY; iy++) {
      G4double y = -halfW + (iy + 0.5) * voxelSize;
      for (G4int iz = 0; iz < nZ; iz++) {
        G4double z = -halfT + (iz + 0.5) * voxelSize;
        nTotal++;
        G4double F = std::sin(k * x) * std::cos(k * y) +
                     std::sin(k * y) * std::cos(k * z) +
                     std::sin(k * z) * std::cos(k * x);
        if (std::abs(F) < threshold) {
          new G4PVPlacement(nullptr, G4ThreeVector(x, y, z), logicVox,
                            "GVoxPLA", fLogicTarget, false, nPlaced, false);
          nPlaced++;
        }
      }
    }
  }

  G4cout << ">>> Gyroid: " << nPlaced << "/" << nTotal << " voxels ("
         << 100.0 * nPlaced / nTotal
         << "% actual infill), voxSize=" << voxelSize / mm << " mm" << G4endl;
}

void DetectorConstruction::ConstructCubicLattice() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solidTarget = new G4Box("Target", halfW, halfW, halfT);
  fLogicTarget = new G4LogicalVolume(solidTarget, fAir, "Target");
  fLogicTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicTarget, "Target",
                    fLogicWorld, false, 0, true);
  MakeTargetRegion(fLogicTarget);

  if (fInfillPercent >= 100.0) {
    G4Box *sf = new G4Box("PLAFill", halfW, halfW, halfT);
    G4LogicalVolume *lf = new G4LogicalVolume(sf, fTargetMaterial, "PLAFill");
    lf->SetUserLimits(new G4UserLimits(0.1 * mm));
    new G4PVPlacement(nullptr, G4ThreeVector(), lf, "PLAFill", fLogicTarget,
                      false, 0, true);
    return;
  }

  G4double targetFrac = fInfillPercent / 100.0;
  G4double wallT = fWallThickness;
  G4double halfWallT = wallT / 2.0;
  G4double beamSigma = 5.0 * mm;

  auto cubicInfill = [&](G4double cc) -> G4double {
    G4int nIn = 0, nProbe = 50000;
    G4int nCX = std::max(1, static_cast<G4int>(fSampleWidth / cc));
    G4int nCZ = std::max(1, static_cast<G4int>(fSampleThickness / cc));
    std::vector<G4double> xWalls, yWalls, zWalls;
    for (G4int i = 0; i <= nCX; i++)
      xWalls.push_back(
          std::clamp(-halfW + i * cc, -halfW + halfWallT, halfW - halfWallT));
    for (G4int j = 0; j <= nCX; j++)
      yWalls.push_back(
          std::clamp(-halfW + j * cc, -halfW + halfWallT, halfW - halfWallT));
    for (G4int k = 0; k <= nCZ; k++)
      zWalls.push_back(
          std::clamp(-halfT + k * cc, -halfT + halfWallT, halfT - halfWallT));
    for (G4int p = 0; p < nProbe; p++) {
      G4double px = G4RandGauss::shoot(0.0, beamSigma);
      G4double py = G4RandGauss::shoot(0.0, beamSigma);
      G4double pz = -halfT + G4UniformRand() * fSampleThickness;
      bool hit = false;
      for (size_t i = 0; i < xWalls.size() && !hit; i++)
        if (std::abs(px - xWalls[i]) < halfWallT)
          hit = true;
      for (size_t j = 0; j < yWalls.size() && !hit; j++)
        if (std::abs(py - yWalls[j]) < halfWallT)
          hit = true;
      for (size_t k = 0; k < zWalls.size() && !hit; k++)
        if (std::abs(pz - zWalls[k]) < halfWallT)
          hit = true;
      if (hit)
        nIn++;
    }
    return static_cast<G4double>(nIn) / nProbe;
  };

  G4double cLo = wallT * 1.5;
  G4double cHi = fSampleWidth / 2.0;
  G4double c = fCellSize;
  for (G4int iter = 0; iter < 25; iter++) {
    G4double frac = cubicInfill(c);
    if (frac < targetFrac)
      cHi = c;
    else
      cLo = c;
    c = (cLo + cHi) / 2.0;
  }

  G4cout << ">>> 3D Grid bisection: c=" << c / mm << " mm for "
         << fInfillPercent << "% infill" << G4endl;

  G4int copyNo = 0;
  G4int nCellsXY = std::max(1, static_cast<G4int>(fSampleWidth / c));
  G4int nCellsZ = std::max(1, static_cast<G4int>(fSampleThickness / c));

  std::vector<G4double> xWallPos, yWallPos;
  for (G4int i = 0; i <= nCellsXY; i++)
    xWallPos.push_back(
        std::clamp(-halfW + i * c, -halfW + halfWallT, halfW - halfWallT));
  for (G4int j = 0; j <= nCellsXY; j++)
    yWallPos.push_back(
        std::clamp(-halfW + j * c, -halfW + halfWallT, halfW - halfWallT));

  for (auto xPos : xWallPos) {
    PlaceWallSlab(halfWallT, halfW, halfT, G4ThreeVector(xPos, 0, 0), nullptr,
                  "CubXWall", copyNo++);
  }

  std::vector<G4double> xBounds = {-halfW};
  for (auto xw : xWallPos) {
    xBounds.push_back(xw - halfWallT);
    xBounds.push_back(xw + halfWallT);
  }
  xBounds.push_back(halfW);
  std::sort(xBounds.begin(), xBounds.end());

  for (auto yPos : yWallPos) {
    for (size_t ix = 0; ix + 1 < xBounds.size(); ix += 2) {
      G4double xLo = xBounds[ix];
      G4double xHi = xBounds[ix + 1];
      G4double hx = (xHi - xLo) / 2.0;
      if (hx < 0.001 * mm)
        continue;
      G4double cx = (xLo + xHi) / 2.0;
      PlaceWallSlab(hx, halfWallT, halfT, G4ThreeVector(cx, yPos, 0), nullptr,
                    "CubYWall", copyNo++);
    }
  }

  std::vector<G4double> yBounds = {-halfW};
  for (auto yw : yWallPos) {
    yBounds.push_back(yw - halfWallT);
    yBounds.push_back(yw + halfWallT);
  }
  yBounds.push_back(halfW);
  std::sort(yBounds.begin(), yBounds.end());

  for (G4int iz = 0; iz <= nCellsZ; iz++) {
    G4double zPos = -halfT + iz * c;
    zPos = std::clamp(zPos, -halfT + halfWallT, halfT - halfWallT);

    for (size_t ix = 0; ix + 1 < xBounds.size(); ix += 2) {
      G4double xLo = xBounds[ix];
      G4double xHi = xBounds[ix + 1];
      G4double hx = (xHi - xLo) / 2.0;
      if (hx < 0.001 * mm)
        continue;
      G4double cx = (xLo + xHi) / 2.0;

      for (size_t iy = 0; iy + 1 < yBounds.size(); iy += 2) {
        G4double yLo = yBounds[iy];
        G4double yHi = yBounds[iy + 1];
        G4double hy = (yHi - yLo) / 2.0;
        if (hy < 0.001 * mm)
          continue;
        G4double cy = (yLo + yHi) / 2.0;

        PlaceWallSlab(hx, hy, halfWallT, G4ThreeVector(cx, cy, zPos), nullptr,
                      "CubZWall", copyNo++);
      }
    }
  }

  G4cout << ">>> 3D Grid: " << copyNo << " walls, cell=" << c / mm
         << " mm, wall=" << wallT / mm << " mm" << G4endl;
}

void DetectorConstruction::ConstructVoronoiLattice() {
  G4double halfW = fSampleWidth / 2.0;
  G4double halfT = fSampleThickness / 2.0;

  G4Box *solidTarget = new G4Box("Target", halfW, halfW, halfT);
  fLogicTarget = new G4LogicalVolume(solidTarget, fAir, "Target");
  fLogicTarget->SetVisAttributes(G4VisAttributes::GetInvisible());
  new G4PVPlacement(nullptr, G4ThreeVector(), fLogicTarget, "Target",
                    fLogicWorld, false, 0, true);
  MakeTargetRegion(fLogicTarget);

  if (fInfillPercent >= 100.0) {
    G4Box *sf = new G4Box("PLAFill", halfW, halfW, halfT);
    G4LogicalVolume *lf = new G4LogicalVolume(sf, fTargetMaterial, "PLAFill");
    lf->SetUserLimits(new G4UserLimits(0.1 * mm));
    new G4PVPlacement(nullptr, G4ThreeVector(), lf, "PLAFill", fLogicTarget,
                      false, 0, true);
    return;
  }

#ifdef USE_CADMESH
  if (!fSTLFile.empty()) {
    G4cout << ">>> Loading Voronoi STL: " << fSTLFile << G4endl;
    auto mesh = CADMesh::TessellatedMesh::FromSTL(fSTLFile);
    mesh->SetScale(mm);
    mesh->SetOffset(G4ThreeVector(-halfW, -halfW, -halfT));
    G4VSolid *solidV = mesh->GetSolid();
    G4LogicalVolume *logicV = new G4LogicalVolume(solidV, fTargetMaterial, "Voronoi");
    logicV->SetUserLimits(new G4UserLimits(0.1 * mm));
    auto *vis = new G4VisAttributes(G4Colour(0.7, 0.3, 0.7, 0.5));
    vis->SetForceSolid(true);
    logicV->SetVisAttributes(vis);
    new G4PVPlacement(nullptr, G4ThreeVector(), logicV, "Voronoi", fLogicTarget,
                      false, 0, true);
    return;
  }
#endif

  G4cout << ">>> Building voxelised Voronoi..." << G4endl;

  G4double voxelSize = 0.4 * mm;
  G4int nX = static_cast<G4int>(fSampleWidth / voxelSize);
  G4int nY = nX;
  G4int nZ = static_cast<G4int>(fSampleThickness / voxelSize);

  while (static_cast<long long>(nX) * nY * nZ > 500000) {
    voxelSize *= 1.2;
    nX = static_cast<G4int>(fSampleWidth / voxelSize);
    nY = nX;
    nZ = static_cast<G4int>(fSampleThickness / voxelSize);
  }

  G4double cellVol = fCellSize * fCellSize * fCellSize;
  G4double sampVol = fSampleWidth * fSampleWidth * fSampleThickness;
  G4int nSeeds = std::max(4, static_cast<G4int>(sampVol / cellVol));

  std::vector<G4ThreeVector> seeds(nSeeds);
  for (G4int s = 0; s < nSeeds; s++) {
    seeds[s] = G4ThreeVector(-halfW + G4UniformRand() * fSampleWidth,
                             -halfW + G4UniformRand() * fSampleWidth,
                             -halfT + G4UniformRand() * fSampleThickness);
  }

  for (G4int iter = 0; iter < 5; iter++) {
    std::vector<G4ThreeVector> centroids(nSeeds, G4ThreeVector());
    std::vector<G4int> counts(nSeeds, 0);
    G4int nProbe = 10000;
    for (G4int p = 0; p < nProbe; p++) {
      G4ThreeVector pt(-halfW + G4UniformRand() * fSampleWidth,
                       -halfW + G4UniformRand() * fSampleWidth,
                       -halfT + G4UniformRand() * fSampleThickness);
      G4double dmin = 1e9;
      G4int closest = 0;
      for (G4int s = 0; s < nSeeds; s++) {
        G4double d = (pt - seeds[s]).mag2();
        if (d < dmin) {
          dmin = d;
          closest = s;
        }
      }
      centroids[closest] += pt;
      counts[closest]++;
    }
    for (G4int s = 0; s < nSeeds; s++) {
      if (counts[s] > 0)
        seeds[s] = centroids[s] / counts[s];
    }
  }

  G4double beamSigma = 5.0 * mm;
  G4double targetFrac = fInfillPercent / 100.0;
  G4double wLo = 0.0, wHi = fCellSize, wallThresh = fWallThickness;
  for (G4int iter = 0; iter < 20; iter++) {
    G4int nIn = 0, nProbe = 20000;
    for (G4int p = 0; p < nProbe; p++) {
      G4ThreeVector pt(G4RandGauss::shoot(0.0, beamSigma),
                       G4RandGauss::shoot(0.0, beamSigma),
                       -halfT + G4UniformRand() * fSampleThickness);
      G4double d1 = 1e9, d2 = 1e9;
      for (G4int s = 0; s < nSeeds; s++) {
        G4double d = (pt - seeds[s]).mag();
        if (d < d1) {
          d2 = d1;
          d1 = d;
        } else if (d < d2) {
          d2 = d;
        }
      }
      if ((d2 - d1) < wallThresh)
        nIn++;
    }
    G4double frac = static_cast<G4double>(nIn) / nProbe;
    if (frac < targetFrac)
      wLo = wallThresh;
    else
      wHi = wallThresh;
    wallThresh = (wLo + wHi) / 2.0;
  }

  G4cout << ">>> Voronoi wallThresh=" << wallThresh / mm << " mm for "
         << fInfillPercent << "% infill" << G4endl;

  G4double halfV = voxelSize / 2.0;
  G4Box *solidVox = new G4Box("VVox", halfV, halfV, halfV);
  G4LogicalVolume *logicVox = new G4LogicalVolume(solidVox, fTargetMaterial, "VVoxPLA");
  logicVox->SetUserLimits(new G4UserLimits(voxelSize * 0.5));
  auto *vis = new G4VisAttributes(G4Colour(0.7, 0.3, 0.7, 0.3));
  vis->SetForceSolid(true);
  logicVox->SetVisAttributes(vis);

  G4int nPlaced = 0, nTotal = 0;
  for (G4int ix = 0; ix < nX; ix++) {
    G4double x = -halfW + (ix + 0.5) * voxelSize;
    for (G4int iy = 0; iy < nY; iy++) {
      G4double y = -halfW + (iy + 0.5) * voxelSize;
      for (G4int iz = 0; iz < nZ; iz++) {
        G4double z = -halfT + (iz + 0.5) * voxelSize;
        nTotal++;
        G4ThreeVector pt(x, y, z);
        G4double d1 = 1e9, d2 = 1e9;
        for (G4int s = 0; s < nSeeds; s++) {
          G4double d = (pt - seeds[s]).mag();
          if (d < d1) {
            d2 = d1;
            d1 = d;
          } else if (d < d2) {
            d2 = d;
          }
        }
        if ((d2 - d1) < wallThresh) {
          new G4PVPlacement(nullptr, G4ThreeVector(x, y, z), logicVox,
                            "VVoxPLA", fLogicTarget, false, nPlaced, false);
          nPlaced++;
        }
      }
    }
  }

  G4cout << ">>> Voronoi: " << nPlaced << "/" << nTotal << " voxels ("
         << 100.0 * nPlaced / nTotal << "%), " << nSeeds << " seeds" << G4endl;
}

void DetectorConstruction::ConstructSDandField() {
}
