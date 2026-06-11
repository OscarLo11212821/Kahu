#include "GameSettingsDialog.h"
#include <QFormLayout>
#include <QVBoxLayout>
#include <QPushButton>
#include <QDialogButtonBox>

GameSettingsDialog::GameSettingsDialog(QWidget *parent) : QDialog(parent) {
    setWindowTitle("Game Settings");
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    QFormLayout *formLayout = new QFormLayout();

    comboWhite = new QComboBox(); comboWhite->addItems({"Human", "Engine"});
    comboBlack = new QComboBox(); comboBlack->addItems({"Human", "Engine"});
    comboBlack->setCurrentIndex(1);

    comboMode = new QComboBox();
    comboMode->addItems({"Fixed Time per Move", "Fixed Depth", "Tournament (Clock)"});
    
    formLayout->addRow("White Player:", comboWhite);
    formLayout->addRow("Black Player:", comboBlack);
    formLayout->addRow("Engine Mode:", comboMode);

    stackParams = new QStackedWidget();

    QWidget *wFixedTime = new QWidget();
    QFormLayout *lFixedTime = new QFormLayout(wFixedTime);
    spinFixedTime = new QSpinBox(); spinFixedTime->setRange(100, 60000); spinFixedTime->setValue(1000); spinFixedTime->setSuffix(" ms");
    lFixedTime->addRow("Time Per Move:", spinFixedTime);
    stackParams->addWidget(wFixedTime);

    QWidget *wFixedDepth = new QWidget();
    QFormLayout *lFixedDepth = new QFormLayout(wFixedDepth);
    spinFixedDepth = new QSpinBox(); spinFixedDepth->setRange(1, 30); spinFixedDepth->setValue(5);
    lFixedDepth->addRow("Search Depth:", spinFixedDepth);
    stackParams->addWidget(wFixedDepth);

    QWidget *wTournament = new QWidget();
    QFormLayout *lTournament = new QFormLayout(wTournament);
    spinBaseTimeMin = new QSpinBox(); spinBaseTimeMin->setRange(1, 180); spinBaseTimeMin->setValue(5); spinBaseTimeMin->setSuffix(" min");
    spinIncrementSec = new QSpinBox(); spinIncrementSec->setRange(0, 60); spinIncrementSec->setValue(3); spinIncrementSec->setSuffix(" sec");
    lTournament->addRow("Base Time:", spinBaseTimeMin);
    lTournament->addRow("Increment:", spinIncrementSec);
    stackParams->addWidget(wTournament);

    formLayout->addRow(stackParams);
    mainLayout->addLayout(formLayout);

    connect(comboMode, &QComboBox::currentIndexChanged, stackParams, &QStackedWidget::setCurrentIndex);

    QDialogButtonBox *btnBox = new QDialogButtonBox(QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    connect(btnBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(btnBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    mainLayout->addWidget(btnBox);
}

PlayerType GameSettingsDialog::getWhitePlayer() const { return comboWhite->currentIndex() == 0 ? PlayerType::Human : PlayerType::Engine; }
PlayerType GameSettingsDialog::getBlackPlayer() const { return comboBlack->currentIndex() == 0 ? PlayerType::Human : PlayerType::Engine; }
EngineMode GameSettingsDialog::getEngineMode() const { return static_cast<EngineMode>(comboMode->currentIndex()); }
int GameSettingsDialog::getFixedTimeMs() const { return spinFixedTime->value(); }
int GameSettingsDialog::getFixedDepth() const { return spinFixedDepth->value(); }
int GameSettingsDialog::getBaseTimeMs() const { return spinBaseTimeMin->value() * 60000; }
int GameSettingsDialog::getIncrementMs() const { return spinIncrementSec->value() * 1000; }