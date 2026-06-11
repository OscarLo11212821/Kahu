#ifndef BOARDWIDGET_H
#define BOARDWIDGET_H

#include <QWidget>
#include <QString>
#include <QMouseEvent>
#include <QPainter>

class BoardWidget : public QWidget {
    Q_OBJECT

public:
    explicit BoardWidget(QWidget *parent = nullptr);
    void updateFromKFEN(const QString &kfen);
    QString getKFEN() const;
    void setSideToMove(char side) { turn = side; }

    void setEditorMode(bool active);
    void setEditorBrush(char type) { editorBrush = type; }
    
    void setCaptured(int w, int b) { capW = w; capB = b; update(); }
    int getCapW() const { return capW; }
    int getCapB() const { return capB; }
    
    void setPVMove(const QString &move) { pvMove = move; update(); }
    void setLastMove(const QString &move) { lastMove = move; update(); }

signals:
    void moveRequested(const QString &move);
    void boardEdited();

protected:
    void paintEvent(QPaintEvent *event) override;
    void mousePressEvent(QMouseEvent *event) override;
    void mouseMoveEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    char boardState[49]; 
    char turn = 'W';
    int capW = 0;
    int capB = 0;
    
    int selectedSquare = -1;
    
    bool bEditorMode = false;
    char editorBrush = 'M'; // 'M' = Move (Drag & Drop)
    
    char draggedPiece = '.';
    QPoint dragPos;
    
    QString pvMove = "";
    QString lastMove = "";

    void drawMarble(QPainter &painter, const QRect &rect, char type);
    void drawArrow(QPainter &painter, const QString &move, const QColor &color);
    QRect getSquareRect(int r, int c) const;
};

#endif // BOARDWIDGET_H