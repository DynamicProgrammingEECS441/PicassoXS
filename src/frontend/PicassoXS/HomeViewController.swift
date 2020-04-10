//
//  ViewController.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/19/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import UIKit
import AVFoundation



struct GlobalVariables {
    static var curInput: UIImage?
    static var curFilter: UIImage?
    static var curResult: UIImage?
    static var FilteredHistory: [MadeImages] = []
    static var Filters: [Filter] = []
    static var Grey1 = UIColor(hex: 0xebebeb)
    static var Grey2 = UIColor(hex: 0xf7f7f6)
    static var White = UIColor(hex: 0xffffff)
    static var UploadMethod = 0
    static var Launched = 0
    static var codeRef = ["document", "francoise","kirchner","munch","roerich","cezanne","gauguin","monet","paul","van-gogh","el-greco","kandinsky","morisot","peploe","pablo","pollock"]
    static var Portrait = 0
    static var Mode = -1
}



class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate,UITableViewDelegate, UITableViewDataSource{
    
    
    @IBOutlet weak var tableview: UITableView!
    
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return 1
    }
    
    func numberOfSections(in tableView: UITableView) -> Int {
        return GlobalVariables.FilteredHistory.count
       }
    
    
    func tableView(_ tableView: UITableView, heightForHeaderInSection section: Int) -> CGFloat {
        return CGFloat(5)
    
    }
    
    
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let curHistory = GlobalVariables.FilteredHistory[indexPath.section]
        let cell = tableView.dequeueReusableCell(withIdentifier: "HomeViewCell") as! HomeViewCell
        cell.setHomeView(Image: curHistory)
        
        //Cell Style
        cell.backgroundColor = UIColor.white
        cell.layer.borderColor = GlobalVariables.White.cgColor
        cell.layer.borderWidth = 1
        cell.layer.cornerRadius = 8
        cell.clipsToBounds = true
        cell.layer.cornerRadius = 50
        cell.layer.shadowOffset = CGSize(width: 0,height: 0)
        cell.layer.shadowOpacity = 0.23
        cell.layer.shadowRadius = 4
        cell.layer.shadowColor = UIColor.black.cgColor
        
        
        return cell
    }
    
    //Screen
    override func viewDidLoad() {
            super.viewDidLoad()
        
        
        if (GlobalVariables.Launched == 0){
            self.loadFilterLibrary()
            let im1 = MadeImages(UIImage(named:"I1")!, UIImage(named:"S6")!, UIImage(named: "R5")!)
            let im2 = MadeImages(UIImage(named:"I4")!, UIImage(named:"S2")!, UIImage(named: "R4")!)
            let im3 = MadeImages(UIImage(named:"I4")!, UIImage(named:"S2")!, UIImage(named: "R2")!)
            let im4 = MadeImages(UIImage(named:"I10")!, UIImage(named:"S6")!, UIImage(named: "R6")!)

            GlobalVariables.FilteredHistory.append(im1)
            GlobalVariables.FilteredHistory.append(im2)
            GlobalVariables.FilteredHistory.append(im3)
            GlobalVariables.FilteredHistory.append(im4)
            GlobalVariables.Launched = 1
        }
   
    
    
    }
    
    
    //使用照相机按键
    @IBAction func Camera(_ sender: Any) {
        print("camera Tapped")
        handleTap()
        if(GlobalVariables.Mode == 1){
            if UIImagePickerController.isSourceTypeAvailable(.camera) {
                let picker = UIImagePickerController()
                    picker.delegate = self
                picker.sourceType = UIImagePickerController.SourceType.camera
                    picker.allowsEditing = false

                picker.mediaTypes = UIImagePickerController.availableMediaTypes(for:.camera)!
                print(picker.mediaTypes)
                self.present(picker, animated: true, completion: nil)
                }
            else {
                    print("can't find camera")
                }
        }
        if(GlobalVariables.Mode == 0){
            print("Portrait Mode")
        }
            
        
        }
    
    let settingsLauncher = SettingsLauncher()
    
    func handleTap(){
        settingsLauncher.showSettings()
    }
    
    
    //使用Library 按键
    @IBAction func Library(_ sender: Any) {
        print("Library Tapped")
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.photoLibrary) {
          let imagePicker = UIImagePickerController()
          imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerController.SourceType.photoLibrary
          imagePicker.allowsEditing = false
          self.present(imagePicker, animated: true, completion: nil)
            
        }
    }
    
    //用户取消拍照
     func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
         dismiss(animated: true, completion: nil)
     }
    
    //用户拍照完成
     func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
         print("\(info)")
         if let image = info[.originalImage] as? UIImage {
            
            //Set Current Input Image
            GlobalVariables.curInput=image
            dismiss(animated: true, completion: nil)
            performSegue(withIdentifier: "Filter", sender: nil)
         }
     }
    
   
    
    
    func loadFilterLibrary(){
        
        let S0 = UIImage(named: "S0")
        GlobalVariables.Filters.append(Filter(S0!,"Add Your Own", "ADDDDDDDYOUROWN"))
        
        let S1 = UIImage(named: "francoise")
        GlobalVariables.Filters.append(Filter(S1!,"Francoise Nielly", "Françoise Nielly was born in Marseille brought up near Cannes and Saint-Tropez and is now living in Paris. In her own way, Françoise Nielly paints the human face in each of his paintings. And she paints it over and over again, with slashes of paint across their face. Moments of life that arise from her paintings are born from a clinch with the canvas. Color is launched like a projectile. Her energy gives each colour tone the intensity of a sound vibration. Paint spots are like music keys, dissonant and noisy just like life itself. While meeting on the canvas these spots of paint harmonize and give to each of the portraits its own soul. Françoise Nielly’s portraits have the ironic beauty of fantasies or that of ghosts, with fascinating and fascinated eyes."))
        
        let S2 = UIImage(named: "kirchner")
        GlobalVariables.Filters.append(Filter(S2!,"Ernst Ludwig Kirchner", "Ernst Ludwig Kirchner (6 May 1880 – 15 June 1938) was a German expressionist painter and printmaker and one of the founders of the artists group Die Brücke or \"The Bridge\", a key group leading to the foundation of Expressionism in 20th-century art. He volunteered for army service in the First World War, but soon suffered a breakdown and was discharged. His work was branded as \"degenerate\" by the Nazis in 1933, and in 1937 more than 600 of his works were sold or destroyed."))
        
        let S3 = UIImage(named: "munch")
        GlobalVariables.Filters.append(Filter(S3!,"Edvard Munch", "Edvard Munch (12 December 1863 – 23 January 1944) was a Norwegian painter. His best known work, The Scream, has become one of the most iconic images of world art.\nTravel brought new influences and outlets. In Paris, he learned much from Paul Gauguin, Vincent van Gogh and Henri de Toulouse-Lautrec, especially their use of colour. In Berlin, he met Swedish dramatist August Strindberg, whom he painted, as he embarked on his major canon The Frieze of Life, depicting a series of deeply-felt themes such as love, anxiety, jealousy and betrayal, steeped in atmosphere"))
        
        let S4 = UIImage(named: "roerich")
        GlobalVariables.Filters.append(Filter(S4!,"Nicholas Roerich", "GNicholas Roerich (October 9, 1874 – December 13, 1947) was a Russian painter, writer, archaeologist, theosophist, philosopher, and public figure, who in his youth was influenced by a movement in Russian society around the spiritual. He was interested in hypnosis and other spiritual practices and his paintings are said to have hypnotic expression.\nRoerich was a dedicated activist for the cause of preserving art and architecture during times of war. He was nominated several times to the long list for the Nobel Peace Prize. The so-called Roerich Pact was signed into law by the United States and most nations of the Pan-American Union in April 1935." ))
        
        let S5 = UIImage(named: "cezanne")
        GlobalVariables.Filters.append(Filter(S5!,"Paul Cézanne", "Paul Cézanne was a French artist and Post-Impressionist painter whose work laid the foundations of the transition from the 19th-century conception of artistic endeavor to a new and radically different world of art in the 20th century.\nCézanne is said to have formed the bridge between late 19th-century Impressionism and the early 20th century's new line of artistic enquiry, Cubism. Cézanne's often repetitive, exploratory brushstrokes are highly characteristic and clearly recognizable. He used planes of colour and small brushstrokes that build up to form complex fields. The paintings convey Cézanne's intense study of his subjects. Both Matisse and Picasso are said to have remarked that Cézanne \"is the father of us all\"."))
        
        let S6 = UIImage(named: "gauguin")
        GlobalVariables.Filters.append(Filter(S6!,"Paul Gauguin","Eugène Henri Paul Gauguin was a French post-Impressionist artist. Unappreciated until after his death, Gauguin is now recognized for his experimental use of color and Synthetist style that were distinct from Impressionism. Toward the end of his life, he spent ten years in French Polynesia, and most of his paintings from this time depict people or landscapes from that region.\nHis work was influential to the French avant-garde and many modern artists, such as Pablo Picasso and Henri Matisse. Gauguin's art became popular after his death, partially from the efforts of art dealer Ambroise Vollard, who organized exhibitions of his work late in his career and assisted in organizing two important posthumous exhibitions in Paris."))
        
        let S7 = UIImage(named: "monet")
        GlobalVariables.Filters.append(Filter(S7!,"Claude Monet","Oscar-Claude Monet was a French painter, a founder of French Impressionist painting and the most consistent and prolific practitioner of the movement's philosophy of expressing one's perceptions before nature, especially as applied to plein air landscape painting. The term \"Impressionism\" is derived from the title of his painting Impression, soleil levant, which was exhibited in 1874 in the first of the independent exhibitions mounted by Monet and his associates as an alternative to the Salon de Paris."))
        
        let S8 = UIImage(named: "paul")
        GlobalVariables.Filters.append(Filter(S8!,"Paul Signac Saint-Tropez", "Paul Signac was born in Paris on 11 November 1863. He followed a course of training in architecture before, at the age of 18, deciding to pursue a career as a painter, after attending an exhibit of Monet's work. He sailed on the Mediterranean Sea, visiting the coasts of Europe and painting the landscapes he encountered. In later years, he also painted a series of watercolors of French harbor cities. Under Seurat's influence he abandoned the short brushstrokes of Impressionism to experiment with scientifically-juxtaposed small dots of pure color, intended to combine and blend not on the canvas, but in the viewer's eye, the defining feature of Pointillism."))
        
        let S9 = UIImage(named: "van-gogh")
        GlobalVariables.Filters.append(Filter(S9!,"Vincent Willem van Gogh", "Vincent Willem van Gogh was a Dutch post-impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade, he created about 2,100 artworks, including around 860 oil paintings, most of which date from the last two years of his life. They include landscapes, still lifes, portraits and self-portraits, and are characterised by bold colours and dramatic, impulsive and expressive brushwork that contributed to the foundations of modern art. He was not commercially successful, and his suicide at 37 came after years of mental illness, depression and poverty."))
        
        let S10 = UIImage(named: "el-greco")
        GlobalVariables.Filters.append(Filter(S10!,"El Greco", "El Greco was born in the Kingdom of Candia, which was at that time part of the Republic of Venice, and the center of Post-Byzantine art. He trained and became a master within that tradition before traveling at age 26 to Venice, as other Greek artists had done. In 1570, he moved to Rome, where he opened a workshop and executed a series of works. During his stay in Italy, El Greco enriched his style with elements of Mannerism and of the Venetian Renaissance taken from a number of great artists of the time, notably Tintoretto. In 1577, he moved to Toledo, Spain, where he lived and worked until his death. In Toledo, El Greco received several major commissions and produced his best-known paintings."))
        
        let S11 = UIImage(named: "kandinsky")
        GlobalVariables.Filters.append(Filter(S11!,"Wassily Kandinsky", "Wassily Wassilyevich Kandinsky was a Russian painter and art theorist. Kandinsky is generally credited as the pioneer of abstract art. Kandinsky began painting studies (life-drawing, sketching and anatomy) at the age of 30. He returned to Moscow in 1914, after the outbreak of World War I. Following the Russian Revolution, Kandinsky \"became an insider in the cultural administration of Anatoly Lunacharsky\" and helped establish the Museum of the Culture of Painting."))
        
        let S12 = UIImage(named: "morisot")
               GlobalVariables.Filters.append(Filter(S12!,"Berthe Morisot", "In 1864, Morisot exhibited for the first time in the highly esteemed Salon de Paris. Sponsored by the government and judged by Academicians, the Salon was the official, annual exhibition of the Académie des beaux-arts in Paris. Her work was selected for exhibition in six subsequent Salons until, in 1874, she joined the \"rejected\" Impressionists in the first of their own exhibitions, which included Paul Cézanne, Edgar Degas, Claude Monet, Camille Pissarro, Pierre-Auguste Renoir and Alfred Sisley. It was held at the studio of the photographer Nadar."))
        
        let S13 = UIImage(named: "peploe")
               GlobalVariables.Filters.append(Filter(S13!,"Samuel Peploe","Samuel John Peploe (pronounced PEP-low; 27 January 1871 – 11 October 1935) was a Scottish Post-Impressionist painter, noted for his still life works and for being one of the group of four painters that became known as the Scottish Colourists."))
        
        let S14 = UIImage(named: "pablo")
                     GlobalVariables.Filters.append(Filter(S14!,"Pablo Picasso","Pablo Ruiz Picasso was a Spanish painter, sculptor, printmaker, ceramicist, stage designer, poet and playwright who spent most of his adult life in France. Regarded as one of the most influential artists of the 20th century, he is known for co-founding the Cubist movement, the invention of constructed sculpture,[5][6] the co-invention of collage, and for the wide variety of styles that he helped develop and explore. Among his most famous works are the proto-Cubist Les Demoiselles d'Avignon (1907), and Guernica (1937), a dramatic portrayal of the bombing of Guernica by the German and Italian airforces during the Spanish Civil War."))
        
        
        let S15 = UIImage(named: "pollock")
                     GlobalVariables.Filters.append(Filter(S15!,"Paul Jackson Pollock ","Paul Jackson Pollock was an American painter and a major figure in the abstract expressionist movement.\nHe was widely noticed for his technique of pouring or splashing liquid household paint onto a horizontal surface ('drip technique'), enabling him to view and paint his canvases from all angles. It was also called 'action painting', since he used the force of his whole body to paint, often in a frenetic dancing style. This extreme form of abstraction divided the critics: some praised the immediacy of the creation, while others derided the random effects. In 2016, Pollock's painting titled Number 17A was reported to have fetched US$200 million in a private purchase."))
    }
    
}


extension UIColor {
    convenience init(hex: Int, alpha: CGFloat = 1.0) {
        let r = CGFloat((hex >> 16) & 0xff) / 255
        let g = CGFloat((hex >> 08) & 0xff) / 255
        let b = CGFloat((hex >> 00) & 0xff) / 255
        self.init(red: r, green: g, blue: b, alpha: alpha)
    }
}



