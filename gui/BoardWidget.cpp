#include "BoardWidget.h"
#include <QPainterPath>
#include <QtMath>

BoardWidget::BoardWidget(QWidget *parent) : QWidget(parent) {
    setMinimumSize(500, 500);
    setMouseTracking(true);
    for (int i = 0; i < 49; ++i) boardState[i] = '.';
}

void BoardWidget::updateFromKFEN(const QString &kfen) {
    QStringList parts = kfen.split(' ');
    QString boardPart = parts.value(0);
    int sq = 0;
    for (QChar c : boardPart) {
        if (c == '/') continue;
        if (c.isDigit()) {
            int empty = c.digitValue();
            for (int i = 0; i < empty; ++i) if(sq < 49) boardState[sq++] = '.';
        } else {
            if(sq < 49) boardState[sq++] = c.toLatin1();
        }
    }
    
    if (parts.size() >= 4) {
        turn = parts[1] == "w" ? 'W' : 'B';
        capW = parts[2].toInt();
        capB = parts[3].toInt();
    }
    
    selectedSquare = -1;
    update();
}

QString BoardWidget::getKFEN() const {
    QString fen;
    for (int r = 0; r < 7; ++r) {
        int emptyCount = 0;
        for (int c = 0; c < 7; ++c) {
            char p = boardState[r * 7 + c];
            if (p == '.') {
                emptyCount++;
            } else {
                if (emptyCount > 0) fen += QString::number(emptyCount);
                emptyCount = 0;
                fen += QChar(p);
            }
        }
        if (emptyCount > 0) fen += QString::number(emptyCount);
        if (r < 6) fen += '/';
    }
    fen += QString(" %1 %2 %3").arg(turn == 'W' ? "w" : "b").arg(capW).arg(capB);
    return fen;
}

void BoardWidget::setEditorMode(bool active) {
    bEditorMode = active;
    selectedSquare = -1;
    draggedPiece = '.';
    pvMove = "";
    lastMove = "";
    update();
}

QRect BoardWidget::getSquareRect(int r, int c) const {
    int sqSize = qMin(width(), height()) / 8;
    int offsetX = (width() - sqSize * 7) / 2;
    int offsetY = (height() - sqSize * 7) / 2;
    
    return QRect(offsetX + c * sqSize, offsetY + r * sqSize, sqSize, sqSize);
}

void BoardWidget::paintEvent(QPaintEvent *) {
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    painter.fillRect(rect(), QColor(160, 105, 60)); 

    int sqSize = qMin(width(), height()) / 8;
    int offsetX = (width() - sqSize * 7) / 2;
    int offsetY = (height() - sqSize * 7) / 2;

    painter.fillRect(offsetX, offsetY, sqSize * 7, sqSize * 7, QColor(222, 184, 135));

    for (int r = 0; r < 7; ++r) {
        for (int c = 0; c < 7; ++c) {
            QRect sqRect = getSquareRect(r, c);
            
            painter.setPen(QPen(Qt::black, 1));
            painter.setBrush(Qt::NoBrush);
            painter.drawRect(sqRect);

            int sqIndex = r * 7 + c;
            
            if (!lastMove.isEmpty() && lastMove.length() >= 2) {
                int lmC = lastMove[0].toLatin1() - 'a';
                int lmR = '7' - lastMove[1].toLatin1();
                if (c == lmC && r == lmR) {
                    painter.fillRect(sqRect, QColor(0, 255, 0, 80));
                }
            }

            if (sqIndex == selectedSquare && !bEditorMode) {
                painter.fillRect(sqRect, QColor(255, 255, 0, 100));
            }
            
            drawMarble(painter, sqRect, boardState[sqIndex]);
        }
    }
    
    if (!pvMove.isEmpty()) drawArrow(painter, pvMove, QColor(0, 200, 255, 180));
    
    // Draw dragged piece on top
    if (bEditorMode && draggedPiece != '.') {
        QRect dRect(dragPos.x() - sqSize/2, dragPos.y() - sqSize/2, sqSize, sqSize);
        drawMarble(painter, dRect, draggedPiece);
    }
}

void BoardWidget::drawMarble(QPainter &painter, const QRect &rect, char type) {
    if (type == '.') return;
    QRect mRect = rect.adjusted(5, 5, -5, -5);
    QRadialGradient grad(mRect.center(), mRect.width() / 2, mRect.topLeft() + QPoint(10, 10));

    if (type == 'W') { grad.setColorAt(0.0, Qt::white); grad.setColorAt(1.0, Qt::lightGray); } 
    else if (type == 'B') { grad.setColorAt(0.0, QColor(80, 80, 80)); grad.setColorAt(1.0, Qt::black); } 
    else if (type == 'R') { grad.setColorAt(0.0, QColor(255, 100, 100)); grad.setColorAt(1.0, Qt::darkRed); }

    painter.setPen(Qt::NoPen);
    painter.setBrush(grad);
    painter.drawEllipse(mRect);
}

void BoardWidget::drawArrow(QPainter &painter, const QString &move, const QColor &color) {
    if (move.length() < 3) return;
    int c = move[0].toLatin1() - 'a';
    int r = '7' - move[1].toLatin1();
    char dir = move[2].toLatin1();

    int destC = c, destR = r;
    if (dir == 'N') destR--; else if (dir == 'S') destR++;
    else if (dir == 'E') destC++; else if (dir == 'W') destC--;

    QRect sq1 = getSquareRect(r, c);
    QRect sq2 = getSquareRect(destR, destC);

    QPoint p1 = sq1.center();
    QPoint p2 = sq2.center();

    painter.setPen(QPen(color, 6, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    painter.drawLine(p1, p2);

    double angle = qAtan2(p2.y() - p1.y(), p2.x() - p1.x());
    QPoint p3 = p2 - QPoint(cos(angle + M_PI / 6) * 15, sin(angle + M_PI / 6) * 15);
    QPoint p4 = p2 - QPoint(cos(angle - M_PI / 6) * 15, sin(angle - M_PI / 6) * 15);
    painter.setBrush(color);
    painter.drawPolygon(QPolygon({p2, p3, p4}));
}

void BoardWidget::mousePressEvent(QMouseEvent *event) {
    int sqSize = qMin(width(), height()) / 8;
    int offsetX = (width() - sqSize * 7) / 2;
    int offsetY = (height() - sqSize * 7) / 2;

    int c = (event->pos().x() - offsetX) / sqSize;
    int r = (event->pos().y() - offsetY) / sqSize;
    
    bool validSq = (r >= 0 && r < 7 && c >= 0 && c < 7);
    int clickedSq = validSq ? (r * 7 + c) : -1;

    if (bEditorMode) {
        if (!validSq) return;
        
        if (editorBrush == 'M') { // Pick up piece for drag & drop
            if (boardState[clickedSq] != '.') {
                draggedPiece = boardState[clickedSq];
                boardState[clickedSq] = '.';
                dragPos = event->pos();
            }
        } else { // Paint brush
            boardState[clickedSq] = editorBrush;
            emit boardEdited();
        }
        update();
        return;
    }

    if (!validSq) {
        selectedSquare = -1;
        update();
        return;
    }

    if (selectedSquare == -1) {
        if (boardState[clickedSq] == turn) selectedSquare = clickedSq;
    } else {
        int selR = selectedSquare / 7;
        int selC = selectedSquare % 7;
        QString dir = "";

        if (c == selC && r == selR - 1) dir = "N";
        else if (c == selC && r == selR + 1) dir = "S";
        else if (r == selR && c == selC + 1) dir = "E";
        else if (r == selR && c == selC - 1) dir = "W";
        
        if (!dir.isEmpty()) {
            char colChar = 'a' + selC;
            char rowChar = '7' - selR; 
            emit moveRequested(QString("%1%2%3").arg(colChar).arg(rowChar).arg(dir));
        }
        selectedSquare = -1;
    }
    update();
}

void BoardWidget::mouseMoveEvent(QMouseEvent *event) {
    if (!bEditorMode) return;

    int sqSize = qMin(width(), height()) / 8;
    int offsetX = (width() - sqSize * 7) / 2;
    int offsetY = (height() - sqSize * 7) / 2;

    int c = (event->pos().x() - offsetX) / sqSize;
    int r = (event->pos().y() - offsetY) / sqSize;

    if (editorBrush == 'M' && draggedPiece != '.') {
        dragPos = event->pos();
        update();
    } else if (editorBrush != 'M' && (event->buttons() & Qt::LeftButton)) {
        if (r >= 0 && r < 7 && c >= 0 && c < 7) {
            int sq = r * 7 + c;
            if (boardState[sq] != editorBrush) {
                boardState[sq] = editorBrush;
                emit boardEdited();
                update();
            }
        }
    }
}

void BoardWidget::mouseReleaseEvent(QMouseEvent *event) {
    if (!bEditorMode || draggedPiece == '.') return;

    int sqSize = qMin(width(), height()) / 8;
    int offsetX = (width() - sqSize * 7) / 2;
    int offsetY = (height() - sqSize * 7) / 2;

    int c = (event->pos().x() - offsetX) / sqSize;
    int r = (event->pos().y() - offsetY) / sqSize;

    if (r >= 0 && r < 7 && c >= 0 && c < 7) {
        int sq = r * 7 + c;
        boardState[sq] = draggedPiece; // Drop piece
    }
    // If dropped out of bounds, piece is effectively deleted.
    
    draggedPiece = '.';
    emit boardEdited();
    update();
}