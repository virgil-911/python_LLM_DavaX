import os
import json
import streamlit as st
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =================== Configuration ===================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# =================== Profanity Filter ===================
class ProfanityFilter:
    def __init__(self):
        """Initialize profanity filter with Romanian and English inappropriate words"""
        # Lista de cuvinte nepotrivite (română și engleză)
        # Am inclus variante comune și prescurtări
        self.inappropriate_words = {
            # Romanian inappropriate words
            'pula', 'pizda', 'muie', 'fut', 'futu', 'fututi', 'futai', 
            'cur', 'cacat', 'rahat', 'mata', 'mortii', 'dracu', 'dracului',
            'plm', 'psd', 'muist', 'muista', 'bulangiu', 'poponar', 'curva',
            'tarfa', 'zdreanta', 'javra', 'jigodie', 'idiot', 'prost', 'tampita',
            'retardat', 'debil', 'cretin', 'imbecil', 'dobitoc', 'bou',
            
            # English inappropriate words
            'fuck', 'shit', 'bitch', 'ass', 'damn', 'bastard', 'dick', 
            'cock', 'pussy', 'whore', 'slut', 'fag', 'retard', 'cunt',
            'nigger', 'nigga', 'asshole', 'motherfucker', 'wtf', 'stfu',
            
            # Common variations and leetspeak
            'f*ck', 'f**k', 'sh*t', 'b*tch', 'a**', 'd*mn', 'p**sy',
            'fvck', 'sh1t', 'b1tch', '@ss', 'pr0st'
        }
        
        # Normalize words for comparison
        self.inappropriate_words = {word.lower() for word in self.inappropriate_words}
    
    def contains_profanity(self, text: str) -> bool:
        """
        Check if text contains inappropriate language
        
        Args:
            text: Text to check
            
        Returns:
            True if profanity detected, False otherwise
        """
        if not text:
            return False
        
        # Convert to lowercase for comparison
        text_lower = text.lower()
        
        # Remove common special characters that might be used to bypass filter
        text_normalized = text_lower
        replacements = {
            '@': 'a', '4': 'a', '3': 'e', '1': 'i', '0': 'o', 
            '5': 's', '7': 't', '*': '', '.': '', '-': '', '_': ''
        }
        for old, new in replacements.items():
            text_normalized = text_normalized.replace(old, new)
        
        # Split into words (handle punctuation)
        import re
        words = re.findall(r'\b\w+\b', text_normalized)
        
        # Check each word
        for word in words:
            if word in self.inappropriate_words:
                return True
            
            # Check for partial matches (word contained within)
            for bad_word in self.inappropriate_words:
                if len(bad_word) >= 4 and bad_word in word:  # Only for longer bad words
                    return True
        
        return False
    
    def get_polite_response(self) -> str:
        """
        Get a polite response when profanity is detected
        
        Returns:
            A polite message asking the user to rephrase
        """
        responses = [
            "Îmi pare rău, dar am detectat limbaj nepotrivit în mesajul tău. Te rog să reformulezi întrebarea într-un mod respectuos.",
            "Pentru a menține o conversație constructivă, te rog să folosești un limbaj adecvat. Cum te pot ajuta cu recomandări de cărți?",
            "Prefer să păstrăm conversația la un nivel profesional. Te rog să reformulezi cererea fără cuvinte ofensatoare.",
            "Sunt aici să te ajut cu recomandări de cărți. Te rog să folosești un limbaj politicos pentru a continua conversația.",
            "Am observat că mesajul tău conține expresii nepotrivite. Hai să ne concentrăm pe găsirea unei cărți perfecte pentru tine!"
        ]
        
        import random
        return random.choice(responses)

# =================== Book Database ===================
# Baza de date cu rezumate scurte pentru ChromaDB
book_summaries_short = [
    {
        "title": "1984",
        "summary": "O poveste distopică despre o societate totalitară controlată prin supraveghere, propagandă și poliția gândirii. Winston Smith, protagonistul, se revoltă în secret împotriva sistemului în căutarea adevărului și libertății.",
        "themes": ["distopie", "totalitarism", "libertate", "supraveghere", "manipulare", "rezistență"]
    },
    {
        "title": "The Hobbit",
        "summary": "Bilbo Baggins, un hobbit liniștit, pornește într-o aventură neașteptată pentru a recupera o comoară de la dragonul Smaug. Descoperă curaj și prietenii durabile într-o lume plină de magie.",
        "themes": ["aventură", "curaj", "prietenie", "magie", "fantezie", "călătorie inițiatică"]
    },
    {
        "title": "To Kill a Mockingbird",
        "summary": "Scout Finch crește în Alabama anilor 1930, învățând despre justiție și prejudecăți rasiale prin experiența tatălui său, avocatul Atticus Finch, care apără un bărbat de culoare acuzat pe nedrept.",
        "themes": ["justiție", "rasism", "inocență", "curaj moral", "copilărie", "prejudecăți"]
    },
    {
        "title": "Pride and Prejudice",
        "summary": "Elizabeth Bennet navighează prin societatea engleză a secolului 19, confruntându-se cu mândria și prejudecățile sale și ale altora, în special în relația cu misteriosul Mr. Darcy.",
        "themes": ["dragoste", "clasă socială", "mândrie", "prejudecată", "familie", "societate"]
    },
    {
        "title": "The Lord of the Rings",
        "summary": "Frodo Baggins pornește într-o misiune epică pentru a distruge Inelul Puterii și a salva Pământul de Mijloc de întuneric. O poveste despre curaj, sacrificiu și puterea prieteniei în fața răului absolut.",
        "themes": ["bine vs rău", "sacrificiu", "prietenie", "putere", "corupție", "eroism", "fantezie epică"]
    },
    {
        "title": "Harry Potter and the Sorcerer's Stone",
        "summary": "Harry Potter descoperă că este vrăjitor și intră în lumea magiei la Hogwarts. Împreună cu prietenii săi, Ron și Hermione, descoperă misterul Pietrei Filozofale și se confruntă cu forțe întunecate.",
        "themes": ["magie", "prietenie", "curaj", "bine vs rău", "descoperire de sine", "școală"]
    },
    {
        "title": "The Great Gatsby",
        "summary": "Jay Gatsby urmărește visul american și dragostea pierdută în New York-ul anilor 1920. O critică a decadenței și superficialității societății americane prin ochii lui Nick Carraway.",
        "themes": ["visul american", "dragoste pierdută", "bogăție", "decadență", "iluzie", "nostalgie"]
    },
    {
        "title": "War and Peace",
        "summary": "Epopeea lui Tolstoi urmărește viețile mai multor familii aristocrate rusești în timpul invaziei napoleoniene. O explorare profundă a naturii umane, războiului, păcii și sensului vieții.",
        "themes": ["război", "pace", "istorie", "dragoste", "familie", "destin", "filozofie"]
    },
    {
        "title": "The Alchemist",
        "summary": "Santiago, un păstor spaniol, călătorește din Spania în Egipt în căutarea unei comori, descoperind în schimb lecții profunde despre urmarea visurilor și ascultarea inimii.",
        "themes": ["destin", "vise", "călătorie spirituală", "dezvoltare personală", "univers", "semne"]
    },
    {
        "title": "Dune",
        "summary": "Pe planeta deșertică Arrakis, Paul Atreides devine liderul unei revolte împotriva unui imperiu galactic corupt. O saga complexă despre politică, religie, ecologie și putere.",
        "themes": ["putere", "religie", "ecologie", "politică", "profeție", "supraviețuire", "science fiction"]
    },
    {
        "title": "The Catcher in the Rye",
        "summary": "Holden Caulfield, un adolescent rebel, rătăcește prin New York după ce a fost exmatriculat din școală. O explorare a alienării adolescentine și a căutării autenticității într-o lume falsă.",
        "themes": ["adolescență", "alienare", "rebeliune", "autenticitate", "inocență", "singurătate"]
    },
    {
        "title": "One Hundred Years of Solitude",
        "summary": "Istoria familiei Buendía de-a lungul a șapte generații în orașul fictiv Macondo. García Márquez țese o poveste magică despre soartă, istorie și natura ciclică a timpului.",
        "themes": ["realism magic", "familie", "solitudine", "destin", "istorie", "timp ciclic"]
    }
]

# Baza de date cu rezumate detaliate pentru tool
book_summaries_detailed = {
    "The Hobbit": (
        "Bilbo Baggins, un hobbit confortabil și fără aventuri, este luat prin surprindere "
        "atunci când vrăjitorul Gandalf și treisprezece pitici conduși de Thorin Oakenshield îl invită "
        "într-o misiune periculoasă de a recupera comoara piticilor păzită de dragonul Smaug. "
        "Pe parcursul călătoriei prin Pământul de Mijloc, Bilbo întâlnește troli, elfi, goblin și "
        "descoperă misteriosul inel care îi conferă invizibilitate. În Pădurea Neagră, se confruntă cu "
        "păianjeni uriași și salvează piticii de la elfii pădurii. La Muntele Singuratic, Bilbo folosește "
        "inteligența pentru a descoperi punctul slab al dragonului. După moartea lui Smaug, trebuie să "
        "medieze între pitici, oameni și elfi pentru a preveni un război. Povestea culminează cu Bătălia "
        "celor Cinci Armate, unde Bilbo descoperă că adevărata comoară nu este aurul, ci prietenia și curajul "
        "pe care le-a găsit în sine. Se întoarce acasă transformat, cu o nouă apreciere pentru aventură și lume."
    ),
    "1984": (
        "Romanul lui George Orwell prezintă o societate distopică în anul 1984, dominată de Partidul "
        "condus de Big Brother. Winston Smith lucrează la Ministerul Adevărului, unde rescrie istoria "
        "conform directivelor Partidului. Trăiește într-o lume unde teleecranele supraveghează constant, "
        "Poliția Gândirii pedepsește crimegândirea, iar limbajul este sistematic simplificat prin Newspeak "
        "pentru a elimina conceptele de rebeliune. Winston începe un jurnal secret și o relație interzisă "
        "cu Julia, colega sa. Împreună visează la rezistență și caută Frăția, o organizație subversivă. "
        "Sunt trădați de O'Brien, pe care îl credeau aliat, și torturați în Ministerul Iubirii. În Camera 101, "
        "Winston este confruntat cu cea mai mare frică - șobolanii - și îl trădează pe Julia. Romanul se "
        "încheie cu Winston complet reeducat, iubindu-l sincer pe Big Brother. Este o avertizare puternică "
        "despre totalitarism, manipulare și pierderea libertății individuale."
    ),
    "To Kill a Mockingbird": (
        "În orașul fictiv Maycomb, Alabama, în anii 1930, Scout Finch povestește copilăria sa alături de "
        "fratele Jem și prietenul Dill. Tatăl lor, Atticus Finch, este un avocat respectat care acceptă să "
        "apere pe Tom Robinson, un bărbat de culoare acuzat pe nedrept de violarea unei femei albe, Mayella Ewell. "
        "În paralel, copiii sunt fascinați de vecinul misterios Boo Radley, despre care circulă povești înfricoșătoare. "
        "Pe măsură ce procesul avansează, Scout și Jem sunt expuși la prejudecățile rasiale ale comunității. "
        "Atticus demonstrează în instanță că Tom este nevinovat, dar juriul îl condamnă oricum. Tom este "
        "împușcat încercând să evadeze. Bob Ewell, tatăl Mayellei, umiliat de proces, atacă copiii lui Atticus, "
        "dar sunt salvați de Boo Radley, care se dovedește a fi un om bun, nu monstrul din povești. "
        "Romanul explorează teme de justiție, curaj moral, pierderea inocenței și puterea distructivă a prejudecăților."
    ),
    "Pride and Prejudice": (
        "Elizabeth Bennet, a doua din cinci surori într-o familie de clasă mijlocie în Anglia rurală, "
        "îl întâlnește pe mândrul Mr. Darcy la un bal local. Prima impresie este dezastruoasă - el pare "
        "arogant și o insultă, refuzând să danseze cu ea. Între timp, sora ei Jane se îndrăgostește de "
        "amabilul Mr. Bingley, prietenul lui Darcy. Elizabeth este fermecată de charmantul ofițer Wickham, "
        "care îi povestește cum Darcy l-a nedreptățit. Darcy o cere în căsătorie pe Elizabeth, care îl "
        "refuză furios, acuzându-l că a separat-o pe Jane de Bingley și l-a nedreptățit pe Wickham. "
        "Într-o scrisoare, Darcy explică adevărul: Wickham este un mincinos care a încercat să fugă cu "
        "sora lui. Elizabeth realizează că a judecat greșit din cauza prejudecăților. Când Lydia, sora "
        "ei cea mică, fuge cu Wickham, Darcy îi salvează în secret familia de rușine. Elizabeth își "
        "recunoaște sentimentele pentru Darcy, iar romanul se încheie cu căsătoriile fericite ale celor "
        "două surori. Este o explorare atemporală a dragostei, claselor sociale și importanței de a privi "
        "dincolo de aparențe."
    ),
    "The Lord of the Rings": (
        "Frodo Baggins moștenește de la unchiul său Bilbo un inel aparent obișnuit, care se dovedește a fi "
        "Inelul Unic creat de Lordul Întunecat Sauron pentru a controla toate celelalte inele ale puterii. "
        "Gandalf îl avertizează că Sauron s-a întors și caută inelul pentru a cuceri Pământul de Mijloc. "
        "La Consiliul din Rivendell, se decide că inelul trebuie distrus în focurile Muntelui Doom unde a "
        "fost creat. Se formează Frăția Inelului: Frodo, Sam, Merry, Pippin, Gandalf, Aragorn, Boromir, "
        "Legolas și Gimli. Frăția se destramă când Boromir încearcă să ia inelul și este ucis apărându-i "
        "pe Merry și Pippin. Frodo și Sam continuă singuri spre Mordor, ghidați ulterior de Gollum. "
        "Aragorn, Legolas și Gimli îi urmăresc pe răpitorii hobbților. Gandalf, reîntors ca Gandalf cel Alb "
        "după lupta cu Balrogul, îi reunește pe eroi pentru bătăliile de la Helm's Deep și Pelennor Fields. "
        "În timp ce armatele se luptă la Porțile Negre ca diversiune, Frodo și Sam reușesc să distrugă "
        "inelul când Gollum cade cu el în lavă. Aragorn devine rege, iar hobbiții se întorc acasă ca eroi. "
        "Este o epopee despre curaj, sacrificiu, prietenie și lupta eternă dintre bine și rău."
    ),
    "Harry Potter and the Sorcerer's Stone": (
        "Harry Potter, un băiat de 11 ani care a crescut cu mătușa și unchiul său abuzivi, descoperă în "
        "ziua de naștere că este vrăjitor și că părinții săi au fost uciși de lordul întunecat Voldemort, "
        "care a încercat să-l omoare și pe el când era bebeluș, lăsându-i doar o cicatrice în formă de fulger. "
        "Hagrid îl duce la Școala de Magie Hogwarts, unde Harry face primii prieteni adevărați - Ron Weasley "
        "și Hermione Granger. Este selectat în Casa Gryffindor și descoperă că are un talent natural la "
        "Quidditch, devenind cel mai tânăr Căutător din ultimul secol. Harry și prietenii săi descoperă că "
        "în școală este ascunsă Piatra Filozofală, care poate produce elixirul vieții. Suspectează că profesorul "
        "Snape încearcă să o fure pentru Voldemort. După o serie de provocări periculoase pentru a proteja "
        "piatra, Harry descoperă că adevăratul trădător este profesorul Quirrell, posedat de Voldemort. "
        "Harry împiedică furtul pietrei, care este apoi distrusă pentru a preveni folosirea ei greșită. "
        "Povestea stabilește temele centrale ale seriei: puterea iubirii și prieteniei împotriva răului, "
        "importanța alegerilor asupra destinului și curajul de a face ceea ce este corect."
    ),
    "The Great Gatsby": (
        "Nick Carraway se mută în Long Island în vara anului 1922, devenind vecin cu misteriosul milionar "
        "Jay Gatsby, care organizează petreceri extravagante în fiecare weekend. Nick descoperă că Gatsby "
        "și verișoara sa, Daisy Buchanan, au avut o relație în urmă cu cinci ani, înainte ca Gatsby să plece "
        "la război. Daisy s-a căsătorit între timp cu bogatul dar brutalul Tom Buchanan. Gatsby a construit "
        "întreaga sa avere prin mijloace îndoielnice doar pentru a o recâștiga pe Daisy, crezând că poate "
        "recrea trecutul. Nick aranjează o reîntâlnire între cei doi, și pentru o vreme pare că visul lui "
        "Gatsby se va împlini. Însă Daisy nu poate abandona securitatea vieții sale. Într-o confruntare "
        "tensionată în New York, Daisy alege să rămână cu Tom. Pe drumul de întoarcere, ea lovește mortal "
        "cu mașina pe Myrtle Wilson, amanta lui Tom, dar Gatsby decide să-și asume vina. George Wilson, "
        "soțul lui Myrtle, îl împușcă pe Gatsby, crezându-l vinovat. La înmormântare, din mulțimea care "
        "venea la petreceri, aproape nimeni nu apare. Nick, dezgustat de superficialitatea și cruzimea "
        "elitei, părăsește New York-ul. Romanul este o critică devastatoare a visului american și a decadenței "
        "morale ascunse sub strălucirea bogăției."
    ),
    "War and Peace": (
        "Romanul monumental al lui Tolstoi urmărește destinele a trei familii aristocrate rusești - Rostov, "
        "Bolkonsky și Bezukhov - pe fundalul invaziei napoleoniene a Rusiei între 1805-1820. Pierre Bezukhov, "
        "moștenitorul nelegitim al unei averi imense, caută sensul vieții prin francmasonerie, filozofie și "
        "dragoste. Se căsătorește cu frumoasa dar infidela Hélène, divorțează și în final găsește fericirea "
        "cu Natasha Rostov. Prințul Andrei Bolkonsky, deziluzionat de viața socială, caută gloria în război, "
        "este rănit la Austerlitz, își pierde soția la naștere, se îndrăgostește de Natasha dar moare din "
        "răni după Borodino. Natasha Rostov evoluează de la o adolescentă romantică la o femeie matură, "
        "aproape fugind cu seducătorul Anatole Kuragin înainte de a-și găsi fericirea cu Pierre. "
        "Nikolai Rostov luptă cu onoare, salvează familia de la ruină financiară și se căsătorește cu "
        "prințesa Maria Bolkonsky. Tolstoi intercalează narațiunea cu reflecții filozofice despre istorie, "
        "liberul arbitru și natura războiului. Descrie în detaliu bătăliile de la Austerlitz și Borodino, "
        "incendierea Moscovei și retragerea dezastruoasă a lui Napoleon. Romanul explorează teme fundamentale: "
        "ce înseamnă să trăiești o viață bună, rolul individului în istorie, natura dragostei și familiei, "
        "și căutarea păcii interioare în mijlocul haosului exterior."
    ),
    "The Alchemist": (
        "Santiago, un tânăr păstor andaluz, are un vis recurent despre o comoară ascunsă la piramidele "
        "egiptene. O țigancă și apoi Melchizedek, regele din Salem, îl încurajează să-și urmeze 'Legenda "
        "Personală'. Vinde oile și pleacă în Africa. În Tanger, este jefuit și lucrează un an la un "
        "negustor de cristale pentru a strânge bani. Învață despre urmărirea visurilor și îmbunătățirea "
        "continuă. Se alătură unei caravane care traversează Sahara, unde întâlnește un englez care "
        "studiază alchimia. La oaza Al-Fayoum, Santiago se îndrăgostește de Fatima și întâlnește "
        "Alchimistul, care devine mentorul său. Alchimistul îl învață să asculte inima sa și să citească "
        "Sufletul Lumii. În drumul spre piramide, sunt capturați de războinici, iar Santiago trebuie să "
        "se transforme în vânt pentru a supraviețui, realizând unitatea sa cu universul. La piramide, "
        "este bătut de hoți care îi spun că au visat despre o comoară într-o biserică spaniolă. Santiago "
        "realizează că adevărata comoară era îngropată de unde a plecat. Se întoarce în Spania, găsește "
        "comoara și se reunește cu Fatima. Romanul transmite că atunci când urmărești cu adevărat visurile, "
        "întregul univers conspiră să te ajute, iar călătoria este la fel de importantă ca destinația."
    ),
    "Dune": (
        "În viitorul îndepărtat, Ducele Leto Atreides primește controlul asupra planetei deșertice Arrakis, "
        "singura sursă din univers a condimentului melanj, esențial pentru călătoriile interstelare și "
        "extinderea conștiinței. Este o capcană a Împăratului și a rivalilor Harkonnen. Leto este trădat "
        "și ucis, dar fiul său Paul și concubina Jessica, membră a ordinului mistic Bene Gesserit, scapă "
        "în deșert. Sunt acceptați de Fremen, nativii Arrakisului, care văd în Paul împlinirea profeției "
        "despre Muad'Dib, mesia lor. Paul descoperă că melanjul îi amplifică puterile de prescience, "
        "văzând multiple viitoruri posibile, inclusiv un jihad galactic în numele său pe care încearcă să-l "
        "evite. Învață căile Fremen, se căsătorește cu Chani și devine liderul lor. Paul îmblânzește viermii "
        "gigantici ai deșertului și conduce Fremen într-o revoltă care răstoarnă Împăratul. Acceptă tronul "
        "imperial dar realizează că a pornit forțe pe care nu le mai poate controla. Romanul explorează "
        "teme complexe: ecologia și importanța mediului, pericolele puterii absolute și ale mesianismului, "
        "politica și religia ca instrumente de control, evoluția umană și conștiința extinsă. Herbert creează "
        "o lume complexă cu culturi, religii și sisteme politice detaliate, făcând din Dune una dintre "
        "operele fundamentale ale science fiction-ului."
    ),
    "The Catcher in the Rye": (
        "Holden Caulfield, 16 ani, povestește evenimentele care au dus la prăbușirea sa nervoasă. După ce "
        "este exmatriculat de la Pencey Prep (a patra școală din care e dat afară), decide să plece mai "
        "devreme în New York înainte de vacanța de Crăciun, evitând să meargă acasă. Rătăcește trei zile "
        "prin oraș, stând la hotel, mergând în baruri și cluburi de noapte, întâlnind diverse persoane pe "
        "care le consideră 'false'. Încearcă să se conecteze cu oameni - o prostituată pe care o plătește "
        "doar să stea de vorbă, o fostă prietenă Sally Hayes cu care are o întâlnire dezastruoasă, fostul "
        "său profesor Mr. Antolini care îl dezamăgește. Este obsedat de soarta rățoilor din Central Park "
        "iarna și de dorința de a fi 'prinsul în secară' care salvează copiii de la căderea de pe o stâncă. "
        "Singura persoană autentică pentru el este sora sa mică, Phoebe. Când ea insistă să fugă cu el, "
        "Holden realizează că nu poate fugi de responsabilități. O duce la carusel în parc și, privind-o "
        "cum se învârte fericită, experimentează un rar moment de bucurie. Narațiunea se încheie cu Holden "
        "într-un sanatoriu, nesigur despre viitor dar aparent mai împăcat. Romanul capturează perfect "
        "alienarea adolescentină, lupta pentru autenticitate într-o lume percepută ca falsă și durerea "
        "tranziției de la inocența copilăriei la complexitatea maturității."
    ),
    "One Hundred Years of Solitude": (
        "José Arcadio Buendía și Úrsula Iguarán, verișori căsătoriți, fondează orașul Macondo după ce fug "
        "din satul natal. Úrsula se teme că vor avea copii cu cozi de porc din cauza consangvinității, o "
        "temere care bântuie familia de-a lungul generațiilor. Primul lor fiu, José Arcadio, fuge cu țiganii; "
        "al doilea, Aureliano, devine colonelul care pornește 32 de războaie civile și le pierde pe toate. "
        "Amaranta, fiica lor, respinge toți pretendentii și țese propriul giulgiu. A doua generație continuă "
        "ciclul: Aureliano Segundo trăiește în exces cu amanta Petra Cotes, José Arcadio Segundo devine "
        "lider sindical și supraviețuiește unui masacru pe care nimeni altcineva nu-l mai ține minte. "
        "A treia generație include pe frumoasa Remedios care levitează către cer, Meme care e trimisă la "
        "mânăstire după o aventură amoroasă, și Amaranta Úrsula care studiază în Europa. În ultima generație, "
        "Aureliano Babilonia descifrează manuscrisele țiganului Melquíades, descoperind că acestea conțin "
        "întreaga istorie a familiei Buendía. În timp ce citește ultimele rânduri, un uragan șterge Macondo "
        "de pe fața pământului, iar profeția copilului cu coadă de porc se împlinește. García Márquez "
        "folosește realismul magic pentru a explora teme de solitudine, destin, natura ciclică a istoriei, "
        "și identitatea latino-americană. Fiecare generație repetă greșelile precedentei, incapabilă să "
        "scape de solitudinea fundamentală a condiției umane."
    )
}

# =================== ChromaDB Setup ===================
class BookRAG:
    def __init__(self):
        """Initialize ChromaDB and OpenAI embedding function"""
        self.client = chromadb.Client()
        
        # OpenAI embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBEDDING_MODEL
        )
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name="book_summaries",
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            self._load_books()
            print("ChromaDB collection created and populated")
        except:
            self.collection = self.client.get_collection(
                name="book_summaries",
                embedding_function=self.embedding_function
            )
            print("Using existing ChromaDB collection")
    
    def _load_books(self):
        """Load books into ChromaDB"""
        documents = []
        metadatas = []
        ids = []
        
        for i, book in enumerate(book_summaries_short):
            # Combine summary and themes for better semantic search
            doc_text = f"{book['summary']} Teme principale: {', '.join(book['themes'])}"
            documents.append(doc_text)
            metadatas.append({
                "title": book["title"],
                "themes": ", ".join(book["themes"])
            })
            ids.append(f"book_{i}")
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_books(self, query: str, n_results: int = 3) -> List[Dict]:
        """Search for books based on semantic similarity"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        books = []
        for i in range(len(results['ids'][0])):
            books.append({
                "title": results['metadatas'][0][i]['title'],
                "themes": results['metadatas'][0][i]['themes'],
                "document": results['documents'][0][i],
                "distance": results['distances'][0][i] if 'distances' in results else None
            })
        
        return books

# =================== Tool Function ===================
def get_summary_by_title(title: str) -> str:
    """
    Returnează rezumatul detaliat pentru un titlu de carte.
    
    Args:
        title: Titlul exact al cărții
        
    Returns:
        Rezumatul detaliat al cărții sau un mesaj de eroare
    """
    if title in book_summaries_detailed:
        return book_summaries_detailed[title]
    else:
        # Încearcă o căutare case-insensitive
        for book_title, summary in book_summaries_detailed.items():
            if book_title.lower() == title.lower():
                return summary
        return f"Nu am găsit un rezumat detaliat pentru '{title}'. Cărțile disponibile sunt: {', '.join(book_summaries_detailed.keys())}"

# Tool definition for OpenAI
tool_definition = {
    "type": "function",
    "function": {
        "name": "get_summary_by_title",
        "description": "Obține un rezumat detaliat pentru o carte specificată prin titlu",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Titlul exact al cărții pentru care se dorește rezumatul"
                }
            },
            "required": ["title"]
        }
    }
}

# =================== Chatbot Class ===================
class BookRecommendationChatbot:
    def __init__(self):
        """Initialize the chatbot with RAG, OpenAI client, and profanity filter"""
        self.rag = BookRAG()
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.profanity_filter = ProfanityFilter()
        self.conversation_history = []
        
    def get_recommendation(self, user_query: str) -> str:
        """
        Get book recommendation based on user query
        
        Args:
            user_query: User's question about books
            
        Returns:
            AI response with book recommendation and detailed summary,
            or polite response if profanity detected
        """
        # Check for profanity first
        if self.profanity_filter.contains_profanity(user_query):
            return self.profanity_filter.get_polite_response()
        
        # If clean, proceed with normal recommendation flow
        # Search for relevant books using RAG
        relevant_books = self.rag.search_books(user_query, n_results=3)
        
        # Prepare context for GPT
        context = "Cărți relevante găsite în baza de date:\n\n"
        for book in relevant_books:
            context += f"Titlu: {book['title']}\n"
            context += f"Teme: {book['themes']}\n"
            context += f"Despre: {book['document']}\n\n"
        
        # System prompt
        system_prompt = """Ești un bibliotecar AI prietenos și cunoscător care recomandă cărți. 
        Folosește informațiile din contextul oferit pentru a recomanda cea mai potrivită carte.
        
        Instrucțiuni:
        1. Analizează cererea utilizatorului și cărțile disponibile
        2. Recomandă UNA dintre cărțile din context care se potrivește cel mai bine
        3. Explică de ce ai ales această carte
        4. IMPORTANT: După recomandare, folosește ÎNTOTDEAUNA funcția get_summary_by_title 
           pentru a obține și afișa rezumatul detaliat
        5. Răspunde în română, într-un ton conversațional și prietenos
        
        Context cu cărți disponibile:
        """ + context
        
        # Create messages for chat
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # First API call - get recommendation
        response = self.client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=[tool_definition],
            tool_choice="auto",
            temperature=0.7
        )
        
        # Process response
        message = response.choices[0].message
        
        # Check if function was called
        if message.tool_calls:
            # Get the function call
            tool_call = message.tool_calls[0]
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            book_title = function_args.get("title", "")
            detailed_summary = get_summary_by_title(book_title)
            
            # Prepare the complete response
            initial_response = message.content if message.content else ""
            
            # Format the final response
            final_response = f"{initial_response}\n\n"
            final_response += "**Rezumat detaliat:**\n\n"
            final_response += f"{detailed_summary}"
            
            return final_response
        else:
            # If no function was called, remind to use it
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": "Te rog folosește funcția get_summary_by_title pentru a oferi rezumatul detaliat al cărții recomandate."})
            
            # Retry with explicit instruction
            retry_response = self.client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                tools=[tool_definition],
                tool_choice={"type": "function", "function": {"name": "get_summary_by_title"}},
                temperature=0.7
            )
            
            retry_message = retry_response.choices[0].message
            if retry_message.tool_calls:
                tool_call = retry_message.tool_calls[0]
                function_args = json.loads(tool_call.function.arguments)
                book_title = function_args.get("title", "")
                detailed_summary = get_summary_by_title(book_title)
                
                final_response = f"{message.content}\n\n"
                final_response += "**Rezumat detaliat:**\n\n"
                final_response += f"{detailed_summary}"
                
                return final_response
            
            return message.content

# =================== Streamlit UI ===================
def main():
    st.set_page_config(
        page_title="Book Recommendation Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("Chatbot pentru Recomandări de Cărți")
    st.markdown("*Powered by OpenAI GPT + ChromaDB RAG*")
    
    # Sidebar with information
    with st.sidebar:
        st.header("Despre Chatbot")
        st.markdown("""
        Acest chatbot folosește:
        - **ChromaDB** pentru căutare semantică
        - **OpenAI GPT** pentru conversație
        - **RAG** (Retrieval Augmented Generation)
        - **Function Calling** pentru rezumate detaliate
        - **Filtru de limbaj** pentru conversații respectuoase
        """)
        
        st.header("Cărți disponibile")
        for book in book_summaries_short:
            st.write(f"• {book['title']}")
        
        st.header("Exemple de întrebări")
        st.markdown("""
        - "Vreau o carte despre prietenie și magie"
        - "Ce recomanzi pentru cineva care iubește poveștile de război?"
        - "Aș vrea ceva despre aventură și curaj"
        - "Caut o carte distopică despre societate"
        - "Îmi place fantasy-ul epic cu prietenie"
        """)
        
        st.header("Reguli de utilizare")
        st.markdown("""
        Te rugăm să folosești un limbaj respectuos și adecvat.
        Mesajele cu conținut ofensator vor fi respinse automat.
        """)
    
    # Initialize session state
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = BookRecommendationChatbot()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Întreabă-mă despre ce fel de carte cauți..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Caut cea mai bună recomandare pentru tine..."):
                try:
                    response = st.session_state.chatbot.get_recommendation(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"A apărut o eroare: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Clear conversation button
    if st.button("Șterge conversația"):
        st.session_state.messages = []
        st.rerun()

# =================== CLI Alternative ===================
def cli_main():
    """Command Line Interface for the chatbot"""
    print("="*60)
    print("CHATBOT PENTRU RECOMANDĂRI DE CĂRȚI")
    print("="*60)
    print("\nBună! Sunt bibliotecarul tău AI. Spune-mi ce fel de carte cauți")
    print("Te rog să folosești un limbaj respectuos în conversație.")
    print("(Scrie 'exit' pentru a ieși)\n")
    
    chatbot = BookRecommendationChatbot()
    
    while True:
        user_input = input("\nTu: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("\nLa revedere! Lectură plăcută!")
            break
        
        if not user_input:
            continue
        
        # Check for profanity before showing "searching" message
        if chatbot.profanity_filter.contains_profanity(user_input):
            print("\nChatbot:", chatbot.profanity_filter.get_polite_response())
            continue
        
        print("\nChatbot: ", end="")
        print("(caut cea mai bună recomandare...)\n")
        
        try:
            response = chatbot.get_recommendation(user_input)
            print(response)
        except Exception as e:
            print(f"\nEroare: {str(e)}")
            print("Te rog verifică API key-ul OpenAI și încearcă din nou.")

# =================== Entry Point ===================
if __name__ == "__main__":
    import sys
    
    # Check for API key
    if not OPENAI_API_KEY:
        print("Te rog setează OPENAI_API_KEY în fișierul .env!")
        print("Exemplu: OPENAI_API_KEY=your-api-key")
        sys.exit(1)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        cli_main()
    else:
        print("Pornesc interfața Streamlit...")
        print("Pentru versiunea CLI, rulează: python app.py --cli")
        main()